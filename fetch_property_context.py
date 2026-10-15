"""Download actual public property context; never backdate its availability."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pandas as pd
import requests

PARCEL_API = "https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Parcel/FeatureServer/0"
CASE_API = "https://data.lacity.org/resource/eagk-wq48.csv"
RENT_API = "https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv"

def fetch_rent_vintage():
    directory = Path("data/property_context")
    directory.mkdir(parents=True,exist_ok=True)
    response = requests.get(RENT_API,timeout=90)
    response.raise_for_status()
    data = pd.read_csv(io.StringIO(response.text),dtype={"RegionName":str})
    data = data[data.State.eq("CA") & data.CountyName.eq("Los Angeles County")]
    if data.empty:
        raise ValueError("No LA County ZIP rent rows; check schema")
    observed = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha256(response.content).hexdigest()
    filename = f"zori_{observed[:10]}_{digest[:12]}.csv.gz"
    data.to_csv(directory/filename,index=False,compression="gzip")
    dates = [c for c in data if len(c)==10 and c[4:5]=="-" and c[:4].isdigit()]
    metadata = dict(source=RENT_API,observed_at=observed,original_payload_sha256=digest,
        file=filename,zip_rows=len(data),latest_month=max(dates),
        limitation="Current revised vintage; not historical release-time data. Excluded from retrospective model fitting.")
    (directory/"rent_provenance.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


def fetch():
    directory = Path("data/property_context")
    directory.mkdir(parents=True, exist_ok=True)
    notices = pd.read_csv("data/cleaned_evictions_final.csv", dtype={"Apn":str}, low_memory=False)
    apns = sorted(set(notices.Apn.fillna("").str.replace(r"\D", "", regex=True).loc[lambda s:s.str.len().eq(10)]))
    fields = ["AIN", "UseCode", "Roll_Year"] + [f"{field}{i}" for i in range(1,6) for field in ["Units", "YearBuilt"]]
    def batch(values):
        where = "AIN IN (" + ",".join("'"+a+"'" for a in values) + ")"
        response = requests.get(PARCEL_API+"/query", params={"f":"json", "where":where,"outFields":",".join(fields),"returnGeometry":"false"}, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "error" in data or data.get("exceededTransferLimit"):
            raise RuntimeError(str(data.get("error", "Parcel query truncated")))
        return [f["attributes"] for f in data.get("features", [])]
    batches = [apns[i:i+150] for i in range(0,len(apns),150)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parcels = pd.DataFrame([row for chunk in pool.map(batch,batches) for row in chunk])
    if parcels.empty or parcels.AIN.duplicated().any():
        raise ValueError("Empty or ambiguous county parcel results")
    parcels.to_csv(directory/"county_parcels_current.csv",index=False)
    response = requests.get(CASE_API, params={"$select":"apn,officialaddress,case_filed_date,closed_date,casetype", "$limit":200000, "$order":"apn,case_filed_date"},timeout=90)
    response.raise_for_status()
    cases = pd.read_csv(io.StringIO(response.text),dtype={"apn":str})
    if len(cases) >= 200000:
        raise ValueError("Enforcement download may be truncated")
    cases = cases[cases.apn.fillna("").str.replace(r"\D", "", regex=True).isin(apns)]
    cases.to_csv(directory/"lahd_cases_current.csv",index=False)
    metadata = dict(observed_at=datetime.now(timezone.utc).isoformat(), parcel_source=PARCEL_API,
        notice_source_sha256=hashlib.sha256(Path("data/cleaned_evictions_final.csv").read_bytes()).hexdigest(),
        parcel_query_fields=fields, parcel_batch_size=150, requested_apns=len(apns), matched_parcels=len(parcels),
        enforcement_source=CASE_API, enforcement_rows=len(cases),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.csv")},
        limitations=["Availability is acquisition time, not historical publication time.",
            "Parcel is not a verified building or owner identifier.", "Units fields describe up to five building lines, not verified rental or occupied units.",
            "Enforcement cases are not necessarily confirmed violations; current closure state cannot be backdated.",
            "Excluded from retrospective model training until dated provenance is established."])
    (directory/"provenance.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ["files", "limitations"]},indent=2))


if __name__ == "__main__":
    import sys
    fetch_rent_vintage() if "--rent-only" in sys.argv else fetch()
