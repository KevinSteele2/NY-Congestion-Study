import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df_daily = pd.read_csv("daily_facility_traffic.csv")

treated_ids = {27, 28}  # Queens Midtown Tunnel, Hugh L. Carey Tunnel
df_daily["group"] = df_daily["facility_id"].apply(lambda x: "treated" if x in treated_ids else "control")

panel = df_daily.groupby(["date", "group"])["total_traffic"].sum().reset_index()
panel["date"] = pd.to_datetime(panel["date"])
pivot = panel.pivot(index="date", columns="group", values="total_traffic").sort_index()

# Parallel-trends check
baseline = pivot.loc["2023-01-01":"2023-12-31"].mean()
smoothed_indexed = pivot.rolling(7).mean().divide(baseline) * 100

fig, ax = plt.subplots(figsize=(12, 5))
smoothed_indexed.loc["2023-01-01":].plot(ax=ax)
ax.axvline(pd.Timestamp("2025-01-05"), color="red", linestyle="--", label="Tolling starts")
ax.set_ylabel("Index (2023 avg = 100), 7-day rolling avg")
ax.legend()
ax.set_title("Treated vs Control Traffic, Smoothed")
plt.savefig("indexed_trends_smoothed.png")
plt.show()

# DiD regression
panel_long = panel.copy()
panel_long["post"] = (panel_long["date"] >= "2025-01-05").astype(int)
panel_long["treated"] = (panel_long["group"] == "treated").astype(int)
panel_long["log_traffic"] = np.log(panel_long["total_traffic"])
panel_long["dow"] = panel_long["date"].dt.dayofweek
panel_long = panel_long[panel_long["date"] >= "2023-01-01"]

model = smf.ols(
    "log_traffic ~ treated + post + treated:post + C(dow)",
    data=panel_long
).fit(cov_type="cluster", cov_kwds={"groups": panel_long["date"].dt.to_period("M")})

print(model.summary())