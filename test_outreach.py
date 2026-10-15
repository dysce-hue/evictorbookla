import unittest
import numpy as np
import pandas as pd
from outreach_metrics import capacity_table, choose_capacity
from test_forecast import fixture
import policy_metrics as pm

class OutreachTests(unittest.TestCase):
    def test_capacity_counts_intervals_and_ties(self):
        ids = np.array([f"A{i}" for i in range(200)])
        y = np.array([1]*40+[0]*160)
        p = np.arange(200,dtype=float)/200
        table = capacity_table(y,p,ids).set_index("Capacity")
        self.assertEqual(table.loc["100 addresses","addresses"],100)
        self.assertEqual(table.loc["Top 5%","addresses"],10)
        self.assertTrue(table.precision_lower_95.le(table.precision).all())
        self.assertTrue(table.precision_upper_95.ge(table.precision).all())
        self.assertTrue(table.recall_lower_95.le(table.recall).all())
        a = capacity_table(y,np.full(200,.2),ids)
        b = capacity_table(y[::-1],np.full(200,.2),ids[::-1])
        pd.testing.assert_frame_equal(a,b)
        empty_positive = capacity_table(np.zeros(200,dtype=int),p,ids)
        self.assertTrue(empty_positive.recall.isna().all())

    def test_capacity_selection_uses_only_earlier_rows(self):
        config = {"family":"test"}
        rows = pd.DataFrame({"config":[str(config)]*4,"origin":["a","a","b","b"],
            "Capacity":["100 addresses","500 addresses"]*2,"precision":[.9,.8,.8,.7],
            "recall":[.1,.3,.15,.4],"requested":[100,500]*2,"population":[1000]*4})
        self.assertEqual(choose_capacity(rows,config),"100 addresses")
        self.assertEqual(choose_capacity(rows.assign(precision=.8),config),"500 addresses")

    def test_added_features_never_read_future(self):
        data = fixture()
        origin = data.Date_Filed.max()-pd.Timedelta(days=90)
        a = pm._forecast_snapshot(data,origin,False)
        b = pm._forecast_snapshot(data[data.Date_Filed.le(origin)],origin,False)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(np.isfinite(a[pm.EXTRA_FORECAST_FEATURES]).all().all())
        self.assertTrue(a.active_week_share_90.between(0,1).all())

if __name__ == "__main__":
    unittest.main()
