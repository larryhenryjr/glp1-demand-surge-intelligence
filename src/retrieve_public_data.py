#!/usr/bin/env python3
"""Reproducible public-data retrieval for GLP-1 demand-surge modeling.

The script retrieves 2018-2025 public CMS SDUD and NADAC records, monthly
Medicaid enrollment, and current openFDA NDC metadata; preserves auditable raw
extracts; builds a state-NDC-quarter panel; and evaluates the pre-specified
next-quarter GLP-1 demand-surge target without fitting a model.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


RUN_DATE = "2026-09-09"
YEARS = list(range(2018, 2026))
BRANDS = [
    "OZEMPIC", "WEGOVY", "RYBELSUS", "TRULICITY", "VICTOZA", "SAXENDA",
    "BYETTA", "BYDUREON", "ADLYXIN", "MOUNJARO", "ZEPBOUND",
]
PRODUCT_SCOPE = {
    "OZEMPIC": ("semaglutide", "GLP-1 receptor agonist"),
    "WEGOVY": ("semaglutide", "GLP-1 receptor agonist"),
    "RYBELSUS": ("semaglutide", "GLP-1 receptor agonist"),
    "TRULICITY": ("dulaglutide", "GLP-1 receptor agonist"),
    "VICTOZA": ("liraglutide", "GLP-1 receptor agonist"),
    "SAXENDA": ("liraglutide", "GLP-1 receptor agonist"),
    "BYETTA": ("exenatide", "GLP-1 receptor agonist"),
    "BYDUREON": ("exenatide", "GLP-1 receptor agonist"),
    "ADLYXIN": ("lixisenatide", "GLP-1 receptor agonist"),
    "MOUNJARO": ("tirzepatide", "dual GIP/GLP-1 receptor agonist"),
    "ZEPBOUND": ("tirzepatide", "dual GIP/GLP-1 receptor agonist"),
}

SDUD_DATASETS = {
    2018: "a1f3598e-fc71-51aa-8560-78e7e1a61b09",
    2019: "daba7980-e219-5996-9bec-90358fd156f1",
    2020: "cc318bfb-a9b2-55f3-a924-d47376b32ea3",
    2021: "eec7fbe6-c4c4-5915-b3d0-be5828ef4e9d",
    2022: "200c2cba-e58d-4a95-aa60-14b99736808d",
    2023: "d890d3a9-6b00-43fd-8b31-fcba4c8e2909",
    2024: "61729e5a-7aa8-448c-8903-ba3e0cd0ea3c",
    2025: "158a1baa-5506-400a-8ec3-97756f0b0536",
}
NADAC_DATASETS = {
    2018: "8de1b213-73c5-552b-b84e-ac795f34d056",
    2019: "76a1984a-6d69-5e4d-86c8-65eb31f0506d",
    2020: "c933dc16-7de9-52b6-8971-4b75992673e0",
    2021: "d5eaf378-dcef-5779-83de-acdd8347d68e",
    2022: "dfa2ab14-06c2-457a-9e36-5cb6d80f8d93",
    2023: "4a00010a-132b-4e4d-a611-543c9521280f",
    2024: "99315a95-37ac-4eee-946a-3c523b4c481e",
    2025: "f38d0706-1239-442c-a3cc-40ef1b686ac0",
}
ENROLLMENT_DATASET = "6165f45b-ca93-5bb5-9d06-db29c692a360"

VALID_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
    "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
    "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

CMS_QUERY = "https://data.medicaid.gov/api/1/datastore/query/{dataset}/0"
OPENFDA_NDC = "https://api.fda.gov/drug/ndc.json"
USER_AGENT = "glp1-demand-surge-intelligence/1.0 (public-data retrieval)"


@dataclass
class FetchResult:
    rows: list[dict[str, Any]]
    expected_count: int
    url: str
    attempts: int
    elapsed_seconds: float


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def json_gz_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)


def json_gz_load(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_url(base: str, params: dict[str, Any]) -> str:
    return base + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str, *, timeout: int = 75, retries: int = 4) -> tuple[dict[str, Any], int, float]:
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload, attempt, time.monotonic() - started
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_error = exc
        except Exception as exc:  # transient network, TLS, JSON, or timeout error
            last_error = exc
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Failed after {retries} attempts: {url}: {last_error}")


def cms_query(dataset: str, conditions: list[tuple[str, str, str]] | None = None,
              *, page_size: int = 5000) -> FetchResult:
    rows: list[dict[str, Any]] = []
    offset = 0
    expected = None
    total_attempts = 0
    total_elapsed = 0.0
    first_url = ""
    while expected is None or offset < expected:
        params: dict[str, Any] = {"limit": page_size, "offset": offset}
        for i, (prop, operator, value) in enumerate(conditions or []):
            params[f"conditions[{i}][property]"] = prop
            params[f"conditions[{i}][operator]"] = operator
            params[f"conditions[{i}][value]"] = value
        url = build_url(CMS_QUERY.format(dataset=dataset), params)
        if not first_url:
            first_url = url
        payload, attempts, elapsed = fetch_json(url)
        total_attempts += attempts
        total_elapsed += elapsed
        page = payload.get("results") or []
        if expected is None:
            expected = int(payload.get("count", len(page)))
        rows.extend(page)
        if not page:
            break
        offset += len(page)
    expected = int(expected or 0)
    if len(rows) != expected:
        raise RuntimeError(f"CMS row count mismatch for {dataset}: expected {expected}, retrieved {len(rows)}")
    return FetchResult(rows, expected, first_url, total_attempts, total_elapsed)


def normalize_ndc11(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    raw = str(value).strip()
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11:
        return digits
    parts = raw.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return parts[0].zfill(5) + parts[1].zfill(4) + parts[2].zfill(2)
    if len(digits) == 10:
        # Unhyphenated 10-digit NDC is ambiguous; do not guess.
        return None
    return digits.zfill(11) if digits and len(digits) < 11 else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def retrieve_sdud(root: Path, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = root / "cache" / "sdud"
    request_rows: list[dict[str, Any]] = []

    def one(year: int, brand: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cache = cache_dir / str(year) / f"{brand.lower()}.json.gz"
        retrieved_at = now_utc()
        if cache.exists():
            payload = json_gz_load(cache)
            rows = payload["rows"]
            meta = payload["meta"]
            meta = {**meta, "cache_used": True}
        else:
            result = cms_query(SDUD_DATASETS[year], [("product_name", "LIKE", f"%{brand}%")])
            rows = result.rows
            meta = {
                "source": "CMS SDUD", "year": year, "brand_filter": brand,
                "dataset_id": SDUD_DATASETS[year], "request_url": result.url,
                "expected_count": result.expected_count, "retrieved_count": len(rows),
                "attempts": result.attempts, "elapsed_seconds": round(result.elapsed_seconds, 3),
                "retrieved_at_utc": retrieved_at, "cache_used": False,
            }
            json_gz_dump(cache, {"meta": meta, "rows": rows})
        for row in rows:
            row["source_brand"] = brand
        return rows, meta

    tasks = [(year, brand) for year in YEARS for brand in BRANDS]
    all_rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(one, y, b): (y, b) for y, b in tasks}
        for future in cf.as_completed(future_map):
            rows, meta = future.result()
            all_rows.extend(rows)
            request_rows.append(meta)
            print(f"SDUD {meta['year']} {meta['brand_filter']}: {len(rows):,}", flush=True)

    raw = pd.DataFrame(all_rows)
    requests = pd.DataFrame(request_rows).sort_values(["year", "brand_filter"])
    for year in YEARS:
        write_csv_gz(root / "raw" / "sdud" / f"sdud_glp1_{year}.csv.gz",
                     raw[pd.to_numeric(raw["year"], errors="coerce") == year].copy())
    requests.to_csv(root / "logs" / "sdud_requests.csv", index=False)
    return raw, requests


def retrieve_enrollment(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = root / "cache" / "enrollment_all.json.gz"
    if cache.exists():
        payload = json_gz_load(cache)
        rows = payload["rows"]
        meta = {**payload["meta"], "cache_used": True}
    else:
        result = cms_query(ENROLLMENT_DATASET)
        rows = result.rows
        meta = {
            "source": "CMS Medicaid & CHIP Enrollment Trends", "dataset_id": ENROLLMENT_DATASET,
            "request_url": result.url, "expected_count": result.expected_count,
            "retrieved_count": len(rows), "attempts": result.attempts,
            "elapsed_seconds": round(result.elapsed_seconds, 3), "retrieved_at_utc": now_utc(),
            "cache_used": False,
        }
        json_gz_dump(cache, {"meta": meta, "rows": rows})
    frame = pd.DataFrame(rows)
    write_csv_gz(root / "raw" / "enrollment" / "medicaid_chip_enrollment_all.csv.gz", frame)
    json_dump(root / "logs" / "enrollment_request.json", meta)
    print(f"Enrollment: {len(frame):,}", flush=True)
    return frame, meta


def retrieve_fda(root: Path, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = root / "cache" / "openfda"
    logs: list[dict[str, Any]] = []

    def one(brand: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cache = cache_dir / f"{brand.lower()}.json.gz"
        if cache.exists():
            payload = json_gz_load(cache)
            return payload["rows"], {**payload["meta"], "cache_used": True}
        params = {"search": f'brand_name:"{brand}"', "limit": 100}
        url = build_url(OPENFDA_NDC, params)
        try:
            payload, attempts, elapsed = fetch_json(url)
            rows = payload.get("results") or []
            expected = int(payload.get("meta", {}).get("results", {}).get("total", len(rows)))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            rows, attempts, elapsed, expected = [], 1, 0.0, 0
        meta = {
            "source": "openFDA NDC Directory", "brand_filter": brand, "request_url": url,
            "expected_count": expected, "retrieved_count": len(rows), "attempts": attempts,
            "elapsed_seconds": round(elapsed, 3), "retrieved_at_utc": now_utc(),
            "cache_used": False,
        }
        json_gz_dump(cache, {"meta": meta, "rows": rows})
        return rows, meta

    records: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=min(workers, 6)) as pool:
        futures = {pool.submit(one, brand): brand for brand in BRANDS}
        for future in cf.as_completed(futures):
            brand = futures[future]
            rows, meta = future.result()
            logs.append(meta)
            for product in rows:
                ingredients = product.get("active_ingredients") or []
                ingredient_names = "; ".join(sorted({str(x.get("name", "")).lower() for x in ingredients if x.get("name")}))
                packages = product.get("packaging") or []
                if not packages:
                    packages = [{}]
                for package in packages:
                    records.append({
                        "source_brand": brand,
                        "brand_name": product.get("brand_name"),
                        "generic_name": product.get("generic_name"),
                        "active_ingredients": ingredient_names,
                        "labeler_name": product.get("labeler_name"),
                        "product_ndc": product.get("product_ndc"),
                        "package_ndc": package.get("package_ndc"),
                        "ndc11": normalize_ndc11(package.get("package_ndc")),
                        "marketing_start_date": package.get("marketing_start_date"),
                        "marketing_end_date": package.get("marketing_end_date"),
                        "listing_expiration_date": product.get("listing_expiration_date"),
                        "finished": product.get("finished"),
                    })
            print(f"openFDA {brand}: {len(rows):,} products", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(root / "raw" / "openfda" / "openfda_glp1_packages.csv", index=False)
    log_frame = pd.DataFrame(logs).sort_values("brand_filter")
    log_frame.to_csv(root / "logs" / "openfda_requests.csv", index=False)
    return frame, log_frame


def retrieve_nadac(root: Path, year_ndcs: dict[int, list[str]], workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = root / "cache" / "nadac"
    logs: list[dict[str, Any]] = []

    def one(year: int, ndc: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cache = cache_dir / str(year) / f"{ndc}.json.gz"
        if cache.exists():
            payload = json_gz_load(cache)
            return payload["rows"], {**payload["meta"], "cache_used": True}
        result = cms_query(NADAC_DATASETS[year], [("ndc", "=", ndc)])
        meta = {
            "source": "CMS NADAC", "year": year, "ndc": ndc,
            "dataset_id": NADAC_DATASETS[year], "request_url": result.url,
            "expected_count": result.expected_count, "retrieved_count": len(result.rows),
            "attempts": result.attempts, "elapsed_seconds": round(result.elapsed_seconds, 3),
            "retrieved_at_utc": now_utc(), "cache_used": False,
        }
        json_gz_dump(cache, {"meta": meta, "rows": result.rows})
        return result.rows, meta

    tasks = [(year, ndc) for year in YEARS for ndc in year_ndcs.get(year, [])]
    all_rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(one, y, n): (y, n) for y, n in tasks}
        completed = 0
        for future in cf.as_completed(future_map):
            rows, meta = future.result()
            for row in rows:
                row["dataset_year"] = meta["year"]
            all_rows.extend(rows)
            logs.append(meta)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"NADAC queries: {completed:,}/{len(tasks):,}", flush=True)
    frame = pd.DataFrame(all_rows)
    for year in YEARS:
        part = frame[pd.to_numeric(frame.get("dataset_year"), errors="coerce") == year].copy() if not frame.empty else frame.copy()
        write_csv_gz(root / "raw" / "nadac" / f"nadac_glp1_{year}.csv.gz", part)
    log_frame = pd.DataFrame(logs).sort_values(["year", "ndc"]) if logs else pd.DataFrame()
    log_frame.to_csv(root / "logs" / "nadac_requests.csv", index=False)
    return frame, log_frame


def clean_sdud(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = raw.copy()
    raw_rows = len(df)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["state", "ndc", "product_name", "utilization_type", "source_brand"]:
        df[col] = df[col].astype("string").str.strip()
    df["state"] = df["state"].str.upper()
    df["source_brand"] = df["source_brand"].str.upper()
    df["ndc11"] = df["ndc"].map(normalize_ndc11)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")
    numeric_cols = [
        "units_reimbursed", "number_of_prescriptions", "total_amount_reimbursed",
        "medicaid_amount_reimbursed", "non_medicaid_amount_reimbursed",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["suppression_used"] = df["suppression_used"].astype("string").str.lower().eq("true")
    df["row_suppressed"] = df["suppression_used"] | df["number_of_prescriptions"].isna()
    exact_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df = df[df["state"].isin(VALID_STATES)].copy()
    invalid_ndc_rows = int(df["ndc11"].isna().sum())
    invalid_period_rows = int((~df["quarter"].isin([1, 2, 3, 4]) | ~df["year"].isin(YEARS)).sum())
    df = df[df["ndc11"].notna() & df["quarter"].isin([1, 2, 3, 4]) & df["year"].isin(YEARS)].copy()
    df["is_mco"] = df["utilization_type"].str.upper().str.startswith("MCO")
    df["mco_rx_component"] = df["number_of_prescriptions"].where(df["is_mco"])
    df["ffs_rx_component"] = df["number_of_prescriptions"].where(~df["is_mco"])

    keys = ["state", "ndc11", "year", "quarter"]
    grouped = df.groupby(keys, dropna=False, as_index=False).agg(
        brand=("source_brand", "first"),
        product_name=("product_name", "first"),
        channel_rows=("utilization_type", "size"),
        channel_types=("utilization_type", lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
        suppression_any=("row_suppressed", "max"),
        prescriptions=("number_of_prescriptions", "sum"),
        units_reimbursed=("units_reimbursed", "sum"),
        total_reimbursed=("total_amount_reimbursed", "sum"),
        medicaid_reimbursed=("medicaid_amount_reimbursed", "sum"),
        non_medicaid_reimbursed=("non_medicaid_amount_reimbursed", "sum"),
        mco_prescriptions=("mco_rx_component", "sum"),
        ffs_prescriptions=("ffs_rx_component", "sum"),
    )
    # Suppressed grains are preserved for audit but their aggregates are not analytically usable.
    measure_cols = ["prescriptions", "units_reimbursed", "total_reimbursed", "medicaid_reimbursed",
                    "non_medicaid_reimbursed", "mco_prescriptions", "ffs_prescriptions"]
    grouped.loc[grouped["suppression_any"], measure_cols] = pd.NA
    duplicate_grains = int(grouped.duplicated(keys).sum())
    scope = pd.DataFrame([
        {"brand": b, "ingredient": PRODUCT_SCOPE[b][0], "drug_class": PRODUCT_SCOPE[b][1]}
        for b in BRANDS
    ])
    grouped = grouped.merge(scope, on="brand", how="left", validate="many_to_one")
    grouped["quarter_index"] = grouped["year"].astype(int) * 4 + grouped["quarter"].astype(int) - 1
    stats = {
        "sdud_raw_channel_rows": raw_rows,
        "sdud_exact_duplicate_rows_removed": exact_dupes,
        "sdud_invalid_ndc_rows_excluded": invalid_ndc_rows,
        "sdud_invalid_period_rows_excluded": invalid_period_rows,
        "sdud_analytic_grains": len(grouped),
        "sdud_suppressed_grains": int(grouped["suppression_any"].sum()),
        "sdud_unsuppressed_grains": int((~grouped["suppression_any"]).sum()),
        "sdud_duplicate_grains_after_aggregation": duplicate_grains,
    }
    return grouped, stats


def clean_enrollment(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["state"] = df["state_abbreviation"].astype("string").str.strip().str.upper()
    df["reporting_period"] = df["reporting_period"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    df["period"] = pd.to_datetime(df["reporting_period"], format="%Y%m", errors="coerce")
    df["enrollment"] = pd.to_numeric(df["total_medicaid_enrollment"], errors="coerce")
    df = df[df["state"].isin(VALID_STATES) & df["period"].notna() & df["period"].dt.year.isin(YEARS)].copy()
    df["is_final"] = df.get("final_report", pd.Series(index=df.index, dtype="object")).astype("string").str.strip().str.lower().isin(["y", "yes", "true", "1"])
    df["is_updated"] = df.get("preliminary_or_updated", pd.Series(index=df.index, dtype="object")).astype("string").str.strip().str.lower().isin(["u", "updated"])
    df["priority"] = df["enrollment"].notna().astype(int) * 4 + df["is_final"].astype(int) * 2 + df["is_updated"].astype(int)
    monthly_dupe_rows = int(df.duplicated(["state", "period"], keep=False).sum())
    conflict = (df.dropna(subset=["enrollment"]).groupby(["state", "period"])["enrollment"].nunique() > 1)
    monthly_conflict_keys = int(conflict.sum())
    df = df.sort_values(["state", "period", "priority", "enrollment"], ascending=[True, True, False, False])
    monthly = df.drop_duplicates(["state", "period"], keep="first").copy()
    monthly["year"] = monthly["period"].dt.year
    monthly["quarter"] = monthly["period"].dt.quarter
    quarterly = monthly.groupby(["state", "year", "quarter"], as_index=False).agg(
        medicaid_enrollment=("enrollment", "mean"),
        enrollment_months=("enrollment", "count"),
        enrollment_month_min=("enrollment", "min"),
        enrollment_month_max=("enrollment", "max"),
    )
    quarterly.loc[quarterly["enrollment_months"] < 2, "medicaid_enrollment"] = pd.NA
    quarterly["quarter_index"] = quarterly["year"] * 4 + quarterly["quarter"] - 1
    stats = {
        "enrollment_filtered_rows": len(df),
        "enrollment_duplicate_month_rows": monthly_dupe_rows,
        "enrollment_conflicting_state_month_keys": monthly_conflict_keys,
        "enrollment_state_months_selected": len(monthly),
        "enrollment_state_quarters": len(quarterly),
        "enrollment_state_quarters_with_2plus_months": int(quarterly["medicaid_enrollment"].notna().sum()),
    }
    return quarterly, stats


def prepare_fda(fda: pd.DataFrame) -> pd.DataFrame:
    if fda.empty:
        return pd.DataFrame(columns=["ndc11", "fda_exact_match", "fda_brand_name", "fda_generic_name", "fda_labeler_name"])
    out = fda[fda["ndc11"].notna()].copy()
    out = out.sort_values(["ndc11", "source_brand", "marketing_start_date"], na_position="last")
    out = out.drop_duplicates("ndc11", keep="last")
    return out[["ndc11", "brand_name", "generic_name", "labeler_name", "marketing_start_date", "marketing_end_date", "listing_expiration_date"]].rename(columns={
        "brand_name": "fda_brand_name", "generic_name": "fda_generic_name", "labeler_name": "fda_labeler_name",
    }).assign(fda_exact_match=True)


def prepare_nadac(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["ndc11", "year", "quarter", "nadac_median", "nadac_latest", "nadac_observations"])
    df = raw.copy()
    df["ndc11"] = df["ndc"].map(normalize_ndc11)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    df["nadac_per_unit"] = pd.to_numeric(df["nadac_per_unit"], errors="coerce")
    df = df[df["ndc11"].notna() & df["as_of_date"].notna() & df["nadac_per_unit"].notna()].copy()
    df["year"] = df["as_of_date"].dt.year
    df["quarter"] = df["as_of_date"].dt.quarter
    df = df.sort_values(["ndc11", "year", "quarter", "as_of_date"])
    grouped = df.groupby(["ndc11", "year", "quarter"], as_index=False).agg(
        nadac_median=("nadac_per_unit", "median"),
        nadac_latest=("nadac_per_unit", "last"),
        nadac_observations=("nadac_per_unit", "size"),
        nadac_pricing_unit=("pricing_unit", "last"),
    )
    return grouped


def add_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["state", "ndc11", "quarter_index"]).copy()
    out["rate_per_100k"] = out["prescriptions"] / out["medicaid_enrollment"] * 100000
    out["units_per_rx"] = out["units_reimbursed"] / out["prescriptions"]
    out["reimbursement_per_rx"] = out["total_reimbursed"] / out["prescriptions"]
    out["medicaid_share"] = out["medicaid_reimbursed"] / out["total_reimbursed"]
    out["mco_share"] = out["mco_prescriptions"] / out["prescriptions"]
    group = out.groupby(["state", "ndc11"], sort=False)
    for lag in [1, 2, 4]:
        value = group["rate_per_100k"].shift(lag)
        idx = group["quarter_index"].shift(lag)
        out[f"rate_lag_{lag}q"] = value.where(out["quarter_index"] - idx == lag)
    out["qoq_rate_growth"] = out["rate_per_100k"] / out["rate_lag_1q"] - 1
    out["yoy_rate_growth"] = out["rate_per_100k"] / out["rate_lag_4q"] - 1
    out["next_rate_per_100k"] = group["rate_per_100k"].shift(-1)
    out["next_quarter_index"] = group["quarter_index"].shift(-1)
    out["next_suppression_any"] = group["suppression_any"].shift(-1)
    out["consecutive_next_quarter"] = out["next_quarter_index"].sub(out["quarter_index"]).eq(1)
    out["next_rate_growth"] = out["next_rate_per_100k"] / out["rate_per_100k"] - 1
    base_eligible = (
        ~out["suppression_any"]
        & out["next_suppression_any"].eq(False)
        & out["consecutive_next_quarter"]
        & out["rate_per_100k"].gt(0)
        & out["next_rate_per_100k"].notna()
        & out["medicaid_enrollment"].notna()
        & out["prescriptions"].ge(50)
    )
    out["target_eligible"] = base_eligible
    for pct in [10, 20, 25, 30]:
        out[f"surge_{pct}pct"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out.loc[base_eligible, f"surge_{pct}pct"] = out.loc[base_eligible, "next_rate_growth"].ge(pct / 100).astype(int)
    out["strict_history_eligible"] = base_eligible & out[["rate_lag_1q", "rate_lag_2q", "rate_lag_4q"]].notna().all(axis=1)
    return out


def build_metrics(panel: pd.DataFrame, raw_sdud: pd.DataFrame, fda: pd.DataFrame,
                  request_logs: dict[str, pd.DataFrame], stats: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    total = len(panel)
    unsuppressed = ~panel["suppression_any"]
    eligible = panel["target_eligible"]
    strict = panel["strict_history_eligible"]
    metrics: list[dict[str, Any]] = []

    def add(metric: str, value: Any, denominator: Any = None, gate: str = "", status: str = "", note: str = "") -> None:
        metrics.append({"metric": metric, "value": value, "denominator": denominator, "gate": gate, "status": status, "note": note})

    for key, value in stats.items():
        add(key, value)
    add("panel_rows", total)
    add("panel_distinct_states", panel["state"].nunique(), 51, "= 51", "PASS" if panel["state"].nunique() == 51 else "FAIL")
    add("panel_distinct_ndcs", panel["ndc11"].nunique())
    add("panel_duplicate_grains", int(panel.duplicated(["state", "ndc11", "year", "quarter"]).sum()), total, "= 0", "PASS" if not panel.duplicated(["state", "ndc11", "year", "quarter"]).any() else "FAIL")
    add("suppression_rate", float(panel["suppression_any"].mean()), total)
    enrollment_cov = float(panel.loc[unsuppressed, "medicaid_enrollment"].notna().mean()) if unsuppressed.any() else 0.0
    add("enrollment_join_coverage", enrollment_cov, int(unsuppressed.sum()), ">= 0.95", "PASS" if enrollment_cov >= .95 else "FAIL")
    fda_cov = float(panel.loc[unsuppressed, "fda_exact_match"].fillna(False).mean()) if unsuppressed.any() else 0.0
    add("openfda_exact_ndc_weighted_coverage", fda_cov, int(unsuppressed.sum()), ">= 0.90", "PASS" if fda_cov >= .90 else "WARN", "Current openFDA exact package-NDC match; manual brand/ingredient scope remains complete.")
    nadac_cov = float(panel.loc[eligible, "nadac_median"].notna().mean()) if eligible.any() else 0.0
    add("nadac_target_row_coverage", nadac_cov, int(eligible.sum()), ">= 0.70 for primary feature", "PASS" if nadac_cov >= .70 else "SENSITIVITY_ONLY")
    add("target_eligible_rows_floor50", int(eligible.sum()), total, ">= 7000", "PASS" if eligible.sum() >= 7000 else "FAIL")
    add("strict_history_eligible_rows", int(strict.sum()), total, ">= 7000 preferred", "PASS" if strict.sum() >= 7000 else "WARN")
    for pct in [10, 20, 25, 30]:
        positives = int(panel.loc[eligible, f"surge_{pct}pct"].sum())
        prevalence = positives / int(eligible.sum()) if eligible.sum() else 0.0
        status = "PASS" if 0.10 <= prevalence <= 0.40 else "WARN"
        add(f"surge_{pct}pct_positive_rows", positives, int(eligible.sum()))
        add(f"surge_{pct}pct_prevalence", prevalence, int(eligible.sum()), "0.10-0.40", status)

    metrics_df = pd.DataFrame(metrics)

    by_year = panel.groupby("year", as_index=False).agg(
        panel_rows=("ndc11", "size"),
        distinct_ndcs=("ndc11", "nunique"),
        states=("state", "nunique"),
        suppressed_grains=("suppression_any", "sum"),
        unsuppressed_grains=("suppression_any", lambda s: int((~s).sum())),
        enrollment_matched=("medicaid_enrollment", lambda s: int(s.notna().sum())),
        fda_exact_matched=("fda_exact_match", lambda s: int(s.fillna(False).sum())),
        target_eligible=("target_eligible", "sum"),
        strict_history_eligible=("strict_history_eligible", "sum"),
        surge_20pct_positive=("surge_20pct", lambda s: int(s.fillna(0).sum())),
    )
    by_year["suppression_rate"] = by_year["suppressed_grains"] / by_year["panel_rows"]
    by_year["surge_20pct_prevalence"] = by_year["surge_20pct_positive"] / by_year["target_eligible"].replace(0, pd.NA)

    by_brand = panel.groupby(["brand", "ingredient"], as_index=False).agg(
        panel_rows=("ndc11", "size"),
        distinct_ndcs=("ndc11", "nunique"),
        suppressed_grains=("suppression_any", "sum"),
        target_eligible=("target_eligible", "sum"),
        strict_history_eligible=("strict_history_eligible", "sum"),
        surge_20pct_positive=("surge_20pct", lambda s: int(s.fillna(0).sum())),
    )
    by_brand["suppression_rate"] = by_brand["suppressed_grains"] / by_brand["panel_rows"]
    by_brand["surge_20pct_prevalence"] = by_brand["surge_20pct_positive"] / by_brand["target_eligible"].replace(0, pd.NA)

    gate_map = metrics_df.set_index("metric")["status"].to_dict()
    core_gates = ["panel_distinct_states", "panel_duplicate_grains", "enrollment_join_coverage", "target_eligible_rows_floor50", "surge_20pct_prevalence"]
    decision = "GO" if all(gate_map.get(g) == "PASS" for g in core_gates) else "CONDITIONAL GO"
    if gate_map.get("target_eligible_rows_floor50") == "FAIL" or gate_map.get("enrollment_join_coverage") == "FAIL":
        decision = "NO-GO WITHOUT DESIGN REVISION"
    summary = {
        "run_date": RUN_DATE,
        "generated_at_utc": now_utc(),
        "decision": decision,
        "panel_rows": int(total),
        "target_eligible_rows": int(eligible.sum()),
        "strict_history_eligible_rows": int(strict.sum()),
        "surge_20pct_prevalence": float(panel.loc[eligible, "surge_20pct"].mean()) if eligible.any() else None,
        "suppression_rate": float(panel["suppression_any"].mean()) if total else None,
        "enrollment_join_coverage": enrollment_cov,
        "openfda_exact_ndc_weighted_coverage": fda_cov,
        "nadac_target_row_coverage": nadac_cov,
        "nadac_role": "primary feature" if nadac_cov >= .70 else "sensitivity analysis only",
        "primary_target": "next-quarter prescriptions per 100,000 Medicaid enrollees grows by at least 20%",
        "current_quarter_floor": 50,
    }
    return metrics_df, by_year, by_brand, summary


def build_manifest(root: Path, files: Iterable[Path], metadata: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for path in sorted(files):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        meta = metadata.get(rel, {})
        rows.append({
            "relative_path": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "created_at_utc": now_utc(),
            **meta,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "source_manifest.csv", index=False)
    return frame


def write_readme(root: Path, summary: dict[str, Any]) -> None:
    text = f"""# GLP-1 2018-2025 feasibility run

Run date: {summary['run_date']}  
Decision: **{summary['decision']}**

This is the pre-notebook retrieval and feasibility package. It retrieves public
CMS State Drug Utilization Data (SDUD), Medicaid enrollment, targeted NADAC
records, and current openFDA NDC metadata; builds the approved state-NDC-quarter
panel; and tests the approved next-quarter surge definition. It does **not** fit
the logistic-regression or XGBoost models.

## Reproduce

```bash
python3 run_feasibility.py --output ./run_output
```

Re-running uses cached responses when present. Delete only a specific cache file
to refresh that request. The request logs record filters, timestamps, expected
counts, retrieved counts, and API URLs. `source_manifest.csv` records SHA-256
checksums for the retained source and analytic files.

## Main outputs

- `processed/analytic_panel_2018_2025.csv.gz`: one row per state, 11-digit NDC,
  year, and quarter, including suppressed grains for audit.
- `processed/modeling_candidates_2018_2025.csv.gz`: rows with a defined
  next-quarter outcome and current-quarter minimum of 50 prescriptions.
- `quality/feasibility_metrics.csv`: gates and decision inputs.
- `quality/by_year.csv` and `quality/by_brand.csv`: coverage and outcome profiles.
- `quality/floor_sensitivity.csv`: sample size and target prevalence under
  alternative current-quarter prescription floors.
- `logs/*_requests.csv`: request-level retrieval evidence.

## Key definitions

- Geography: 50 states plus DC; aggregate and non-state codes are excluded.
- Grain: state x 11-digit NDC x calendar quarter.
- Channels: FFS and managed care are aggregated only when every contributing
  channel is unsuppressed. A grain with any suppression remains in the audit
  panel but is ineligible for the primary model.
- Rate: prescriptions / mean monthly Medicaid enrollment x 100,000.
- Target: next-quarter rate growth >=20%, requiring consecutive quarters and a
  current-quarter floor of 50 prescriptions. Sensitivities: 10%, 25%, and 30%.
- NADAC: included as a primary feature only when target-row coverage is >=70%;
  otherwise retained for sensitivity analysis.

## Important limitation

CMS suppresses small counts, so the usable panel overrepresents higher-volume
state-product quarters. The design is predictive, not causal, and a successful
model would not establish that any feature caused demand growth.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "run_output")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    root = args.output.resolve()
    for sub in ["raw/sdud", "raw/enrollment", "raw/openfda", "raw/nadac", "processed", "quality", "logs", "cache"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    run_started = now_utc()
    raw_sdud, sdud_logs = retrieve_sdud(root, args.workers)
    raw_enrollment, enrollment_meta = retrieve_enrollment(root)
    raw_fda, fda_logs = retrieve_fda(root, args.workers)

    panel, sdud_stats = clean_sdud(raw_sdud)
    year_ndcs = {
        year: sorted(panel.loc[panel["year"].astype(int) == year, "ndc11"].dropna().astype(str).unique())
        for year in YEARS
    }
    raw_nadac, nadac_logs = retrieve_nadac(root, year_ndcs, args.workers)
    enrollment, enrollment_stats = clean_enrollment(raw_enrollment)
    fda_dim = prepare_fda(raw_fda)
    nadac_q = prepare_nadac(raw_nadac)

    panel = panel.merge(enrollment[["state", "year", "quarter", "medicaid_enrollment", "enrollment_months"]],
                        on=["state", "year", "quarter"], how="left", validate="many_to_one")
    panel = panel.merge(fda_dim, on="ndc11", how="left", validate="many_to_one")
    panel["fda_exact_match"] = panel["fda_exact_match"].eq(True)
    panel = panel.merge(nadac_q, on=["ndc11", "year", "quarter"], how="left", validate="many_to_one")
    panel = add_panel_features(panel)

    impossible = (
        panel.loc[~panel["suppression_any"], "prescriptions"].lt(0).sum()
        + panel.loc[~panel["suppression_any"], "total_reimbursed"].lt(0).sum()
        + panel.loc[~panel["suppression_any"], "medicaid_enrollment"].le(0).sum()
    )
    all_stats = {**sdud_stats, **enrollment_stats, "negative_or_nonpositive_core_measure_rows": int(impossible)}
    metrics, by_year, by_brand, summary = build_metrics(
        panel, raw_sdud, raw_fda,
        {"sdud": sdud_logs, "fda": fda_logs, "nadac": nadac_logs},
        all_stats,
    )

    write_csv_gz(root / "processed" / "analytic_panel_2018_2025.csv.gz", panel)
    candidates = panel[panel["target_eligible"]].copy()
    write_csv_gz(root / "processed" / "modeling_candidates_2018_2025.csv.gz", candidates)
    metrics.to_csv(root / "quality" / "feasibility_metrics.csv", index=False)
    by_year.to_csv(root / "quality" / "by_year.csv", index=False)
    by_brand.to_csv(root / "quality" / "by_brand.csv", index=False)
    floor_rows = []
    base = panel.sort_values(["state", "ndc11", "quarter_index"]).copy()
    grouped = base.groupby(["state", "ndc11"], sort=False)
    next_rate = grouped["rate_per_100k"].shift(-1)
    next_index = grouped["quarter_index"].shift(-1)
    next_suppressed = grouped["suppression_any"].shift(-1)
    next_growth = next_rate / base["rate_per_100k"] - 1
    for floor in [0, 25, 50, 100, 250]:
        eligible_floor = (
            ~base["suppression_any"] & next_suppressed.eq(False)
            & next_index.sub(base["quarter_index"]).eq(1)
            & base["rate_per_100k"].gt(0) & next_rate.notna()
            & base["prescriptions"].ge(floor)
        )
        strict_floor = eligible_floor & base[["rate_lag_1q", "rate_lag_2q", "rate_lag_4q"]].notna().all(axis=1)
        floor_rows.append({
            "prescription_floor": floor,
            "target_eligible_rows": int(eligible_floor.sum()),
            "strict_history_rows": int(strict_floor.sum()),
            "surge_20pct_positive": int(next_growth[eligible_floor].ge(.20).sum()),
            "surge_20pct_prevalence": float(next_growth[eligible_floor].ge(.20).mean()),
        })
    pd.DataFrame(floor_rows).to_csv(root / "quality" / "floor_sensitivity.csv", index=False)
    by_quarter = base.groupby(["year", "quarter"], as_index=False).agg(
        panel_rows=("ndc11", "size"),
        suppressed_grains=("suppression_any", "sum"),
        target_eligible=("target_eligible", "sum"),
        strict_history_eligible=("strict_history_eligible", "sum"),
        surge_20pct_positive=("surge_20pct", lambda s: int(s.fillna(0).sum())),
    )
    by_quarter["period"] = by_quarter["year"].astype(str) + " Q" + by_quarter["quarter"].astype(str)
    by_quarter["suppression_rate"] = by_quarter["suppressed_grains"] / by_quarter["panel_rows"]
    by_quarter["surge_20pct_prevalence"] = by_quarter["surge_20pct_positive"] / by_quarter["target_eligible"].replace(0, pd.NA)
    by_quarter.to_csv(root / "quality" / "by_quarter.csv", index=False)
    json_dump(root / "quality" / "feasibility_summary.json", summary)
    json_dump(root / "quality" / "run_metadata.json", {
        "run_started_utc": run_started, "run_completed_utc": now_utc(),
        "python": sys.version, "pandas": pd.__version__, "run_date": RUN_DATE,
        "brands": BRANDS, "years": YEARS,
    })
    pd.DataFrame([
        {"brand": b, "ingredient": PRODUCT_SCOPE[b][0], "drug_class": PRODUCT_SCOPE[b][1], "include": True}
        for b in BRANDS
    ]).to_csv(root / "product_scope.csv", index=False)
    write_readme(root, summary)

    retained = [p for p in root.rglob("*") if p.is_file() and "cache" not in p.parts and p.name != "source_manifest.csv"]
    build_manifest(root, retained, {})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
