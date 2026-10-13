"""Rent CPI retrieval with a public-download fallback and no credential logging."""
import io
import re

import pandas as pd
import requests

RENT_SERIES = "CUSR0000SEHA"


def fetch_rent_cpi(api_key="", session=None):
    session = session or requests.Session()
    notes = []
    if api_key and not re.fullmatch(r"[a-z0-9]{32}", api_key.strip()):
        notes.append("Configured FRED key has an invalid format; a 32-character key is required.")
    elif api_key:
        try:
            response = session.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": RENT_SERIES, "api_key": api_key.strip(),
                        "file_type": "json", "observation_start": "2015-01-01"},
                timeout=30,
            )
            response.raise_for_status()
            frame = pd.DataFrame(response.json()["observations"])[["date", "value"]]
            frame["date"] = pd.to_datetime(frame["date"])
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            return frame.dropna(), "FRED API", notes
        except (requests.RequestException, ValueError, KeyError):
            notes.append("FRED API unavailable or key rejected; trying the public CSV.")
    try:
        response = session.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                               params={"id": RENT_SERIES, "cosd": "2015-01-01"}, timeout=45)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        frame = frame.rename(columns={frame.columns[0]: "date", RENT_SERIES: "value"})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame[["date", "value"]].dropna(), "FRED public CSV", notes
    except (requests.RequestException, ValueError, KeyError, IndexError):
        notes.append("FRED public download is currently unavailable; retry after the cache expires.")
        return pd.DataFrame(columns=["date", "value"]), "Unavailable", notes
