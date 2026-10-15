import pandas as pd
import json
import plotly.express as px
import streamlit as st

def render_outreach(result):
    if "outreach_evaluation" not in result:
        st.info("Rebuild this forecast using the current model version to obtain capacity evaluations.")
        return
    meta = result["metadata"]
    st.subheader("Precision and recall at outreach capacity")
    st.caption("Same eligible test population for all models. This is retrospective evaluation—not a genuinely unseen future test. Choosing fewer addresses can raise precision while missing more future notices.")
    validation = result["outreach_validation"]
    selected = validation[validation.config.eq(str(meta["selected_parameters"]))]
    means = selected.groupby("Capacity",as_index=False).agg(precision=("precision","mean"),recall=("recall","mean"),periods=("origin","nunique"))
    st.write(f"Earlier-validation capacity choice: **{meta.get('selected_capacity') or 'Unavailable'}**. Rule: highest mean precision, then recall, among capacities supported in every earlier period. This is an illustrative precision-first rule—not an optimized outreach budget or a 95% guarantee.")
    st.dataframe(means.rename(columns={"precision":"Mean validation precision","recall":"Mean validation recall","periods":"Periods"}).style.format({"Mean validation precision":"{:.1%}","Mean validation recall":"{:.1%}"}),hide_index=True,width="stretch")
    table = result["outreach_evaluation"].copy()
    fig = px.line(table.sort_values(["Model","recall","precision"]),x="recall",y="precision",color="Model",markers=True,hover_data=["Capacity","addresses","true_positives","false_positives"])
    fig.update_layout(xaxis=dict(title="Recall: share of future-notice addresses reached",tickformat=".0%",range=[0,1]),yaxis=dict(title="Precision: share of flagged addresses with a notice",tickformat=".0%",range=[0,1]),legend=dict(orientation="h"),height=470)
    st.plotly_chart(fig,width="stretch")
    st.caption("Each point is one outreach capacity, not the full precision–recall curve.")
    labels = {"addresses":"Flagged addresses","true_positives":"Received another notice","false_positives":"No observed notice","precision":"Precision","recall":"Recall","precision_lower_95":"Precision 95% lower","precision_upper_95":"Precision 95% upper"}
    display = table[["Model","Capacity",*labels]].rename(columns=labels)
    st.dataframe(display.style.format({c:"{:.1%}" for c in display if "Precision" in c or c=="Recall"}),hide_index=True,width="stretch")
    st.caption("Ties are resolved by a fixed SHA-256 address ordering independent of outcomes. Absolute capacities are clipped when population is smaller and are excluded from selection if unsupported. Precision and recall intervals are approximate 95% Wilson intervals; they ignore spatial dependence, reporting errors and selection uncertainty. Multiple capacity comparisons are not independent.")
    st.download_button("Download capacity evaluation",table.to_csv(index=False),"outreach_capacity_evaluation.csv","text/csv")
    st.download_button("Download earlier capacity-validation scores",validation.to_csv(index=False),"outreach_capacity_validation.csv","text/csv")
    st.warning(f"No genuinely future outcomes have been evaluated here. Freeze a future forecast and protocol before observing outcomes, then wait 90 days plus a justified reporting buffer. The displayed forecast origin is {pd.Timestamp(meta['forecast_as_of']):%B %d, %Y}.")
    with st.expander("Prepare a future evaluation protocol"):
        today = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None).normalize()
        next_quarter = (today.to_period("Q")+1).start_time-pd.Timedelta(days=1)
        origin = st.date_input("Future forecast origin",value=next_quarter.date(),min_value=(today+pd.Timedelta(days=1)).date(),key="prospective_origin")
        protocol = dict(status="Protocol only; predictions and outcomes not yet frozen/evaluated", registered_at=pd.Timestamp.now(tz="UTC").isoformat(),
            forecast_origin=str(origin),horizon_days=90,model_version=meta["model_version"],fixed_parameters=meta["selected_parameters"],
            feature_names=meta.get("feature_names"),capacity_rule=meta.get("selected_capacity"),
            evaluation_rules=["At origin, fit fixed parameters using only mature outcomes and information actually available by that date.",
                "Save predictions, intervals, eligibility list, source checksums and timestamps without overwriting.",
                "Choose and justify a reporting-delay buffer from earlier captures before evaluation.",
                "Evaluate after horizon plus buffer; retain a later untouched period.",
                "Do not use this evaluation to tune the model, capacity or threshold and still call it untouched."])
        st.download_button("Download future evaluation protocol",json.dumps(protocol,indent=2),"future_evaluation_protocol.json","application/json")
        st.caption("This prepares a protocol—not a scheduled job or a frozen future prediction. No future performance is claimed. Source data and predictions must be captured when the origin actually arrives.")
