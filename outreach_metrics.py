"""Fixed-capacity precision/recall; select only on earlier periods."""
import hashlib
import numpy as np
import pandas as pd


def wilson(hits, total):
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    q = hits / total
    center = (q + z*z/(2*total)) / (1+z*z/total)
    margin = z*np.sqrt(q*(1-q)/total+z*z/(4*total*total))/(1+z*z/total)
    return max(0.,min(q,center-margin)), min(1.,max(q,center+margin))


def capacity_table(target, probability, addresses):
    y, p = np.asarray(target,dtype=int), np.asarray(probability,dtype=float)
    if not len(y):
        return pd.DataFrame()
    if len(y) != len(p) or not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
        raise ValueError("Capacity metrics require aligned binary outcomes and finite scores")
    ids = [hashlib.sha256(str(a).encode()).hexdigest() for a in addresses]
    if len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ValueError("Capacity evaluation requires one unique address per row")
    order = np.lexsort((np.asarray(ids),-p))
    options = [("100 addresses",100),("500 addresses",500),("1,000 addresses",1000)]
    options += [(f"Top {int(s*100)}%",max(1,int(np.ceil(s*len(y))))) for s in [.01,.05,.10]]
    positives = int(y.sum())
    rows = []
    for label, requested in options:
        k = min(len(y),requested)
        hits = int(y[order[:k]].sum())
        lower,upper = wilson(hits,k)
        rlow,rhigh = wilson(hits,positives)
        rows.append(dict(Capacity=label, addresses=k, requested=requested, true_positives=hits,
            false_positives=k-hits, total_positives=positives, precision=hits/k,
            recall=hits/positives if positives else np.nan,
            precision_lower_95=lower, precision_upper_95=upper, recall_lower_95=rlow, recall_upper_95=rhigh,
            population=len(y), population_share=k/len(y)))
    return pd.DataFrame(rows)


def choose_capacity(validation, config):
    """Equal-period mean precision, then recall; exclude clipped absolute K.

    This illustrative rule prioritizes precision, not an optimal policy budget.
    Evaluation outcomes are deliberately not an argument.
    """
    rows = validation[validation.config.eq(str(config))].copy()
    rows = rows[rows.requested.le(rows.population)]
    means = rows.groupby("Capacity",as_index=False).agg(precision=("precision","mean"), recall=("recall","mean"), periods=("origin","nunique"))
    means = means[means.periods.eq(rows.origin.nunique())]
    if means.empty:
        return None
    return means.sort_values(["precision","recall","Capacity"],ascending=[False,False,True]).iloc[0].Capacity
