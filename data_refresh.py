"""Safely refresh EvictorBookLA from LAHD's published eviction CSV."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SOURCE_PAGE = "https://housing.lacity.gov/residents/renters/eviction-notices-filed"
SOURCE_CSV = "https://housingapp.lacity.org/cpradocs/Eviction%20Notice%20Workbook.xlsx.csv"
DEFAULT_OUTPUT = Path("data/cleaned_evictions_final.csv")
METADATA_PATH = Path("data/refresh_metadata.json")
GEOCODE_CACHE = Path("data/geocode_cache.csv")
EXPECTED_SOURCE = {
    "Notice Date", "Date Received", "Eviction Category", "Cause", "Address",
    "City", "Zip", "Bedroom", "Notice Type", "Apn", "Council District", "Rent Owed",
}


class RefreshError(RuntimeError):
    pass


class StaleSourceError(RefreshError):
    pass


def address_key(series: pd.Series) -> pd.Series:
    return (
        series.fillna("").astype(str).str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )


def parse_source_content(content: bytes) -> tuple[pd.DataFrame, str]:
    """LAHD sometimes serves an XLSX workbook from its .csv URL."""
    if content.startswith(b"PK\x03\x04"):
        frames = []
        with pd.ExcelFile(io.BytesIO(content), engine="openpyxl") as workbook:
            for sheet in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet, dtype={"Zip": str, "Apn": str})
                frame.columns = frame.columns.astype(str).str.strip()
                if EXPECTED_SOURCE.issubset(frame.columns):
                    frames.append(frame)
        if not frames:
            raise RefreshError("No worksheet contains the required LAHD notice columns.")
        return pd.concat(frames, ignore_index=True), "xlsx"
    frame = pd.read_csv(io.BytesIO(content), dtype={"Zip": str, "Apn": str}, encoding="utf-8-sig")
    frame.columns = frame.columns.astype(str).str.strip()
    return frame, "csv"


def download_source(session: requests.Session, archive_directory=None) -> tuple[pd.DataFrame, dict]:
    response = session.get(SOURCE_CSV, timeout=180)
    response.raise_for_status()
    if len(response.content) < 1000:
        raise RefreshError("LAHD download was unexpectedly small.")
    frame, source_format = parse_source_content(response.content)
    missing = sorted(EXPECTED_SOURCE - set(frame.columns))
    if missing:
        raise RefreshError(f"LAHD source schema changed; missing columns: {missing}")
    if archive_directory is not None:
        from model_data import archive_download
        archive_download(response.content, archive_directory, SOURCE_CSV)
    return frame, {
        "url": SOURCE_CSV,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "bytes": len(response.content),
        "format": source_format,
    }


def normalize_source(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["Zip"] = frame["Zip"].fillna("").str.replace(r"\.0$", "", regex=True).str.zfill(5)
    frame["Apn"] = frame["Apn"].fillna("").str.replace(r"\.0$", "", regex=True)
    frame["full_address"] = (
        frame["Address"].fillna("").str.strip() + ", "
        + frame["City"].fillna("Los Angeles").str.strip() + ", " + frame["Zip"]
    )
    frame["full_address_clean"] = address_key(frame["full_address"])
    frame["repeat_count"] = frame.groupby("full_address_clean")["full_address_clean"].transform("size")
    frame["Date Received"] = pd.to_datetime(frame["Date Received"], format="mixed", errors="coerce").dt.strftime("%m-%d-%Y")
    frame["Notice Date"] = pd.to_datetime(frame["Notice Date"], format="mixed", errors="coerce").dt.strftime("%m-%d-%Y")
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame.insert(0, "row_id", range(1, len(frame) + 1))
    return frame


def attach_known_coordinates(frame: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    result = frame.copy()
    result["lat"] = np.nan
    result["long"] = np.nan
    if existing is None or not {"lat", "long"}.issubset(existing.columns):
        return result
    old = existing.dropna(subset=["lat", "long"]).copy()
    old["_key"] = address_key(old.get("full_address", old.get("full_address_clean")))
    by_address = old.drop_duplicates("_key").set_index("_key")[["lat", "long"]]
    result[["lat", "long"]] = result["full_address_clean"].map(by_address["lat"]).to_frame().join(
        result["full_address_clean"].map(by_address["long"]).rename("long")
    ).set_axis(result.index)

    # Use APN only when its existing coordinates are unambiguous.
    coordinate_counts = old.groupby("Apn")[["lat", "long"]].nunique()
    valid_apns = coordinate_counts[(coordinate_counts["lat"] == 1) & (coordinate_counts["long"] == 1)].index
    valid_apns = valid_apns[~valid_apns.astype(str).isin(["", "0", "nan", "None"])]
    by_apn = old[old["Apn"].isin(valid_apns)].drop_duplicates("Apn").set_index("Apn")[["lat", "long"]]
    missing = result["lat"].isna()
    result.loc[missing, "lat"] = result.loc[missing, "Apn"].map(by_apn["lat"])
    result.loc[missing, "long"] = result.loc[missing, "Apn"].map(by_apn["long"])
    return result


def _load_cache() -> pd.DataFrame:
    if not GEOCODE_CACHE.exists():
        return pd.DataFrame(columns=["full_address_clean", "lat", "long", "matched_address"])
    cache = pd.read_csv(GEOCODE_CACHE).drop_duplicates("full_address_clean", keep="last")
    cache["lat"] = pd.to_numeric(cache["lat"], errors="coerce")
    cache["long"] = pd.to_numeric(cache["long"], errors="coerce")
    return cache


def apply_geocode_cache(frame: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if cache.empty:
        return result
    cache = cache.copy()
    cache["lat"] = pd.to_numeric(cache["lat"], errors="coerce")
    cache["long"] = pd.to_numeric(cache["long"], errors="coerce")
    lookup = cache.dropna(subset=["lat", "long"]).set_index("full_address_clean")[["lat", "long"]]
    missing = result["lat"].isna()
    result.loc[missing, "lat"] = result.loc[missing, "full_address_clean"].map(lookup["lat"])
    result.loc[missing, "long"] = result.loc[missing, "full_address_clean"].map(lookup["long"])
    return result


def census_batch_geocode(session: requests.Session, addresses: pd.DataFrame) -> pd.DataFrame:
    """Geocode unique addresses in Census batches; failures remain cached as unmatched."""
    endpoint = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
    outputs = []
    unique = addresses[["full_address_clean", "Address", "City", "Zip"]].drop_duplicates("full_address_clean")
    for offset in range(0, len(unique), 9000):
        batch = unique.iloc[offset:offset + 9000].reset_index(drop=True)
        upload = io.StringIO()
        for idx, row in batch.iterrows():
            street = str(row["Address"] or "").replace(",", " ")
            city = str(row["City"] or "Los Angeles").replace(",", " ")
            upload.write(f'{idx},"{street}","{city}",CA,{row["Zip"]}\n')
        files = {"addressFile": ("addresses.csv", upload.getvalue(), "text/csv")}
        data = {"benchmark": "Public_AR_Current", "vintage": "Current_Current"}
        response = session.post(endpoint, files=files, data=data, timeout=300)
        response.raise_for_status()
        parsed = pd.read_csv(io.StringIO(response.text), header=None, dtype=str, on_bad_lines="skip")
        for record in parsed.itertuples(index=False, name=None):
            idx = int(record[0])
            status = record[2] if len(record) > 2 else "No_Match"
            matched_address = record[4] if len(record) > 4 else ""
            coordinates = record[5] if len(record) > 5 else ""
            lon = lat = np.nan
            if status == "Match" and isinstance(coordinates, str) and "," in coordinates:
                try:
                    lon, lat = map(float, coordinates.split(",", 1))
                except ValueError:
                    pass
            outputs.append({
                "full_address_clean": batch.iloc[idx]["full_address_clean"],
                "lat": lat, "long": lon, "matched_address": matched_address,
            })
    return pd.DataFrame(outputs)


def validate(candidate: pd.DataFrame, existing: pd.DataFrame | None = None) -> dict:
    parsed_dates = pd.to_datetime(candidate["Date Received"], format="%m-%d-%Y", errors="coerce")
    mapped = candidate[["lat", "long"]].notna().all(axis=1)
    checks = {
        "rows": int(len(candidate)),
        "unique_addresses": int(candidate["full_address_clean"].nunique()),
        "valid_date_rows": int(parsed_dates.notna().sum()),
        "mapped_rows": int(mapped.sum()),
        "mapped_percent": round(float(mapped.mean() * 100), 2),
        "latest_received": parsed_dates.max().date().isoformat() if parsed_dates.notna().any() else None,
        "existing_rows": int(len(existing)) if existing is not None else None,
        "new_rows_vs_existing": int(len(candidate) - len(existing)) if existing is not None else None,
    }
    if len(candidate) == 0 or parsed_dates.notna().mean() < 0.99 or mapped.mean() < 0.75:
        raise RefreshError(f"Candidate dataset failed validation: {checks}")
    if existing is not None and len(candidate) < len(existing) * 0.8:
        raise RefreshError(f"Candidate unexpectedly shrank by more than 20%: {checks}")
    if existing is not None:
        old_dates = pd.to_datetime(existing["Date Received"], format="mixed", errors="coerce")
        old_months = set(old_dates.dropna().dt.strftime("%Y-%m"))
        new_months = set(parsed_dates.dropna().dt.strftime("%Y-%m"))
        if old_months - new_months or parsed_dates.max() < old_dates.max():
            raise StaleSourceError(f"Published download ends {parsed_dates.max().date()}, while saved data ends {old_dates.max().date()}, or historical months are absent. Newer saved data retained.")
    checks["monthly_received_counts"] = {str(k): int(v) for k, v in parsed_dates.dt.strftime("%Y-%m").value_counts().sort_index().items()}
    return checks


def refresh(output: Path = DEFAULT_OUTPUT, dry_run: bool = False, geocode: bool = True) -> dict:
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1, status_forcelist=[429, 502, 503, 504])))
    source, source_meta = download_source(session, None if dry_run else output.parent / "source_archives")
    existing = pd.read_csv(output, dtype={"Zip": str, "Apn": str}) if output.exists() else None
    candidate = attach_known_coordinates(normalize_source(source), existing)
    cache = _load_cache()
    candidate = apply_geocode_cache(candidate, cache)
    geocoded_now = 0
    if geocode:
        missing_unique = candidate.loc[candidate["lat"].isna(), ["full_address_clean", "Address", "City", "Zip"]].drop_duplicates("full_address_clean")
        if not missing_unique.empty:
            additions = census_batch_geocode(session, missing_unique)
            geocoded_now = int(additions["lat"].notna().sum())
            cache = pd.concat([cache, additions], ignore_index=True).drop_duplicates("full_address_clean", keep="last")
            candidate = apply_geocode_cache(candidate, cache)
    try:
        checks = validate(candidate, existing)
    except StaleSourceError as error:
        metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        metadata.update({"checked_at": datetime.now(timezone.utc).isoformat(),
                         "status": "retained_newer_local_data", "message": str(error),
                         "last_attempt_source": source_meta, "dry_run": dry_run})
        if not dry_run:
            METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        return metadata
    metadata = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "source_page": SOURCE_PAGE, "source_csv": source_meta,
        "checks": checks, "geocoded_new_unique_addresses": geocoded_now,
        "dry_run": dry_run,
        "status": "updated",
    }
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            shutil.copy2(output, output.with_suffix(".previous.csv"))
        temporary = output.with_suffix(".next.csv")
        candidate.to_csv(temporary, index=False)
        os.replace(temporary, output)
        if not cache.empty:
            cache.to_csv(GEOCODE_CACHE, index=False)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-geocode", action="store_true")
    args = parser.parse_args()
    print(json.dumps(refresh(args.output, args.dry_run, not args.no_geocode), indent=2))


if __name__ == "__main__":
    main()
