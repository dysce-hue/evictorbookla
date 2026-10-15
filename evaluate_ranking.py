"""Reproduce the actual-data ranking experiment without starting Streamlit."""
import json
import hashlib
from pathlib import Path
import pandas as pd
from policy_metrics import prepare_notices, predictive_building_forecast

if __name__ == "__main__":
    result = predictive_building_forecast(prepare_notices(pd.read_csv("data/cleaned_evictions_final.csv",low_memory=False)))
    output = Path("data/ranking_evaluation")
    output.mkdir(parents=True,exist_ok=True)
    for key in ["validation_trials","outreach_validation","outreach_evaluation","model_comparison"]:
        result[key].to_csv(output/f"{key}.csv",index=False)
    metadata = result["metadata"]
    metadata["notice_source_sha256"] = hashlib.sha256(Path("data/cleaned_evictions_final.csv").read_bytes()).hexdigest()
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2,default=str))
    print(result["model_comparison"].to_string(index=False),flush=True)
    print("PARAMETERS",metadata["selected_parameters"],"CAPACITY",metadata["selected_capacity"],flush=True)
    print(result["validation_trials"].to_string(index=False),flush=True)
    print(result["outreach_evaluation"].to_string(index=False),flush=True)
