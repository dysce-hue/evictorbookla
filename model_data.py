"""Prospective source archives and strict point-in-time enrichment helpers.

No optional enrichment is silently added to the production predictor.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import pandas as pd


def archive_download(content, directory, source, captured_at=None):
    """Keep exact payload bytes once per hash, and every download observation.

    The first capture is a left-censored baseline, never historical first arrival.
    Local archives must be backed up; cloud ephemeral disks are not durable.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(captured_at or datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        raise ValueError("Capture timestamps must include timezone")
    digest = hashlib.sha256(content).hexdigest()
    payload = directory / f"{digest}.payload.gz"
    if not payload.exists():
        temporary = directory / f"{digest}.{os.getpid()}.tmp"
        temporary.write_bytes(gzip.compress(content, mtime=0))
        os.replace(temporary, payload)
    with sqlite3.connect(directory / "captures.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS captures (captured_at TEXT, sha256 TEXT, source TEXT, bytes INTEGER)")
        db.execute("INSERT INTO captures VALUES (?,?,?,?)", (stamp.tz_convert("UTC").isoformat(), digest, source, len(content)))
    return dict(captured_at=stamp.isoformat(), sha256=digest, bytes=len(content))


def capture_inventory(directory):
    path = Path(directory) / "captures.sqlite"
    if not path.exists():
        return pd.DataFrame(columns=["captured_at", "sha256", "source", "bytes"])
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return pd.read_sql_query("SELECT * FROM captures ORDER BY captured_at", db)


def reporting_delay_report(directory):
    """Observed first-seen minus received-date, excluding baseline identities.

    Full source-field hashes are proxies, not stable notice IDs. Corrections
    can look like new records; exact duplicate notices cannot be distinguished.
    Dates measure observed availability, not the agency's publication time.
    """
    from data_refresh import EXPECTED_SOURCE, parse_source_content
    inventory = capture_inventory(directory).drop_duplicates("sha256")
    seen, rows = set(), []
    latest_received, stale_count = None, 0
    for position, capture in enumerate(inventory.itertuples(index=False)):
        content = gzip.decompress((Path(directory) / f"{capture.sha256}.payload.gz").read_bytes())
        if hashlib.sha256(content).hexdigest() != capture.sha256:
            raise ValueError("Archive payload checksum mismatch")
        frame, _ = parse_source_content(content)
        received_max = pd.to_datetime(frame["Date Received"],format="mixed",errors="coerce").max()
        if latest_received is not None and pd.notna(received_max) and received_max < latest_received:
            stale_count += 1
            continue
        if pd.notna(received_max):
            latest_received = received_max
        canonical = frame.reindex(columns=sorted(EXPECTED_SOURCE)).fillna("").astype(str)
        for field in ["Date Received", "Notice Date"]:
            canonical[field] = pd.to_datetime(canonical[field], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        for field in ["Apn", "Zip", "Council District", "Rent Owed"]:
            canonical[field] = canonical[field].str.replace(r"\.0$", "", regex=True)
        hashes = canonical.apply(lambda row: hashlib.sha256(json.dumps(row.tolist(), ensure_ascii=False).encode()).hexdigest(), axis=1)
        new = ~hashes.isin(seen)
        if position > 0:
            first_seen = pd.Timestamp(capture.captured_at).tz_convert("America/Los_Angeles").tz_localize(None).normalize()
            received = pd.to_datetime(canonical.loc[new, "Date Received"], errors="coerce")
            delay = (first_seen - received).dt.days
            for key, date, days in zip(hashes[new], received, delay):
                if pd.notna(days) and days >= 0:
                    rows.append(dict(record_fingerprint=key, first_seen=capture.captured_at,
                                     received_date=date.date().isoformat(), observed_delay_days=int(days)))
        seen.update(hashes)
    report = pd.DataFrame(rows, columns=["record_fingerprint", "first_seen", "received_date", "observed_delay_days"]).drop_duplicates("record_fingerprint")
    report.attrs["excluded_stale_snapshots"] = stale_count
    return report


def point_in_time_attributes(addresses, attributes, origin):
    """Review-only import: exact verified building IDs and two date gates.

    Reject ambiguous versions instead of multiplying notice rows. Preserve
    missing matches; never infer one unit or an unregulated building from null.
    """
    required = {"building_id", "effective_at", "available_at", "source_url"}
    if not required.issubset(attributes):
        raise ValueError(f"Missing fields: {sorted(required-set(attributes))}")
    data = attributes.copy()
    for column in ["effective_at", "available_at"]:
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
    cutoff = pd.Timestamp(origin)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if data[list(required)].isna().any().any() or data.source_url.astype(str).str.strip().eq("").any():
        raise ValueError("Building identity, source and availability dates cannot be missing")
    if data.building_id.astype(str).str.strip().eq("").any():
        raise ValueError("Building IDs cannot be empty")
    if "rental_units" in data:
        units = pd.to_numeric(data.rental_units, errors="raise")
        if ((units.notna()) & ((units <= 0) | (units % 1 != 0))).any():
            raise ValueError("Known rental unit counts must be positive integers")
    data = data[(data.effective_at <= cutoff) & (data.available_at <= cutoff)]
    if "year_built" in data:
        year = pd.to_numeric(data.year_built, errors="raise")
        if (year.notna() & ((year <= 0) | (year > cutoff.year) | (year % 1 != 0))).any():
            raise ValueError("Known construction years must be positive integers no later than origin")
    if data.duplicated(["building_id", "effective_at", "available_at"]).any():
        raise ValueError("Ambiguous duplicate building versions")
    latest = data.sort_values(["effective_at", "available_at"]).drop_duplicates("building_id", keep="last")
    return addresses.merge(latest, on="building_id", how="left", validate="many_to_one")
