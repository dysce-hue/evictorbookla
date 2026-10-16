"""Acquire a bounded official selected-area RSO cohort, never call it citywide."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import requests
from property_panel import save_panel

SOURCE = "https://services1.arcgis.com/tzwalEyxl2rpamKs/ArcGIS/rest/services/LAHD_RSO_May2025_AssessorParcels/FeatureServer/17"

def fetch():
    def query(**params):
        r = requests.get(SOURCE+"/query",params={"f":"json",**params},timeout=40)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        return data
    count = query(where="1=1",returnCountOnly="true")["count"]
    rows = []
    for offset in range(0,count,2000):
        data = query(where="1=1",outFields="OBJECTID,APN,RSO_Year,Address,City,Zip,RSO_Units,Council_District",returnGeometry="false",orderByFields="OBJECTID ASC",resultOffset=offset,resultRecordCount=2000)
        rows.extend(feature["attributes"] for feature in data.get("features",[]))
    final_count = query(where="1=1",returnCountOnly="true")["count"]
    frame = pd.DataFrame(rows)
    if len(frame) != count or count != final_count or frame.OBJECTID.duplicated().any():
        raise ValueError("Source changed or pagination incomplete; do not use this capture")
    directory = Path("data/property_panel")
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/"selected_area_rso_inventory.csv.gz"
    frame.to_csv(path,index=False,compression="gzip")
    observed = datetime.now(timezone.utc).isoformat()
    metadata = dict(source=SOURCE,observed_at=observed,source_rows=count,
        scope="Official planning selected-area RSO layer (selectionTOD); NOT a citywide rental-property universe",
        fields=list(frame),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        unknowns=["Historic membership dates","Citywide/non-RSO rental coverage","Verified building IDs","Historic owners","Court outcomes"])
    (directory/"provenance.json").write_text(json.dumps(metadata,indent=2))
    save_panel("data/cleaned_evictions_final.csv",path,observed,directory)

if __name__ == "__main__":
    import sys
    if "--build-only" in sys.argv:
        directory = Path("data/property_panel")
        meta = json.loads((directory/"provenance.json").read_text())
        save_panel("data/cleaned_evictions_final.csv",directory/"selected_area_rso_inventory.csv.gz",meta["observed_at"],directory)
    else:
        fetch()
