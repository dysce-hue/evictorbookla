"""Bounded descriptive panel, explicitly separate from the existing predictor."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

DIRECTORY = Path("data/property_panel")

def render_property_panel():
    st.subheader("Property-month pilot: selected-area RSO parcels")
    if not (DIRECTORY/"audit.json").exists():
        st.info("Panel not built on this host. Run fetch_rso_panel.py to acquire the public selected-area inventory and build it.")
        return
    audit = json.loads((DIRECTORY/"audit.json").read_text())
    provenance = json.loads((DIRECTORY/"provenance.json").read_text())
    st.warning("This is NOT a citywide rental-property universe. It uses an official selected-area RSO layer. Current inventory membership is not verified historically, so no rows are eligible to train a new citywide model.")
    st.caption("Full pilot cohort; the sidebar address, cause, date and council-district filters do not apply. The existing address-recurrence predictor is unchanged.")
    cards = st.columns(3)
    cards[0].metric("Reference parcels",f"{audit['universe_apns']:,}")
    cards[1].metric("No observed notice in extract",f"{audit['universe_apns_without_observed_notice']:,}")
    cards[2].metric("Training-eligible panel rows",f"{audit['training_eligible_rows']:,}")
    st.write(f"{audit['panel_rows']:,} APN-month rows over {audit['monthly_origins']} month-end origins. Inventory observed {audit['inventory_observed_at'][:10]}; notice records span {audit['source_first_received']}–{audit['source_last_received']}.")
    st.caption("An APN identifies a parcel—not a verified building, unit or landlord. Zero observed notices does not establish no court case, no eviction, or complete reporting.")
    summary = pd.read_csv(DIRECTORY/"monthly_summary.csv",parse_dates=["origin"])
    valid = summary[summary.month_inside_extract.eq(True)]
    if not valid.empty:
        chart = px.line(valid,x="origin",y="observed_notices",markers=True)
        chart.update_layout(height=360,xaxis_title="Month-end",yaxis_title="Observed notice records",showlegend=False)
        st.plotly_chart(chart,width="stretch")
        st.caption("Only calendar months wholly inside the extract date bounds are plotted. These bounds do not prove reporting completeness. Counts describe the selected current-inventory cohort, not all Los Angeles properties.")
    st.markdown("### Keep notice stages separate")
    st.caption("LAHD enforcement-case records are not UD court filings. Date Received is the agency notice-received date, not a court filing or lockout date. No stage transition is inferred merely because a prior stage exists.")
    stages = pd.DataFrame({"Stage":["LAHD notice received","UD court filing","Judgment","Writ of possession","Lockout","Historical owner/network"],
        "Evidence":["Observed notice extract","Not available","Not available","Not available","Not available","Not available"],
        "Panel treatment":["Observed counts; mature 90-day observed-notice labels only","Unknown/null, not zero","Unknown/null, not zero","Unknown/null, not zero","Unknown/null, not zero","Unknown/null, not inferred from names"]})
    st.dataframe(stages,hide_index=True,width="stretch")
    st.markdown("### Matching and exposure checks")
    checks = pd.DataFrame({"Check":["Notice rows matched to this subset","Notice rows with invalid/missing APN","Notice rows outside this subset","Inventory APNs with ambiguous unit counts","Inventory APNs with missing/invalid unit counts"],
        "Rows / APNs":[audit["notice_rows_matched_to_subset"],audit["notice_invalid_apn_rows"],audit["notice_rows_outside_subset"],audit["ambiguous_unit_apns"],audit["inventory_missing_unit_apns"]]})
    st.dataframe(checks,hide_index=True,width="stretch")
    st.caption("Exact normalized APNs only. No fuzzy/nearest matches, invented match probabilities or owner-network links. Conflicting unit values remain unknown; they are not summed. Current RSO units are context, not historical rental/occupied-unit exposure.")
    st.markdown(f"[Official selected-area source]({provenance['source']})")
    st.download_button("Download panel audit",json.dumps(audit,indent=2),"parcel_month_audit.json","application/json")
    st.download_button("Download monthly summary",summary.to_csv(index=False),"parcel_month_summary.csv","text/csv")
    with st.expander("Download full pilot and inspect the data contract"):
        st.write("The panel includes past-only 30/90/180/365-day notice counts and recency, zero-observed months, extract-boundary flags, separate unknown court stages and a disabled training label. Current-inventory membership and past source availability are not established. This is a descriptive research table, not a new validated predictor.")
        st.download_button("Download parcel-month pilot (Parquet)",(DIRECTORY/"parcel_month_pilot.parquet").read_bytes(),"parcel_month_pilot.parquet","application/octet-stream")
    st.info("Next requirement: a dated, complete rental-property universe and historical membership/exposure records. Court stages require separately obtained and verified court/outcome linkage; landlord features require dated ownership entities. No causal conclusion or 95% predictive guarantee is established by this pilot.")
