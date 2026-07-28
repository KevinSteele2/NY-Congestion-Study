# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df_daily = pd.read_csv("daily_facility_traffic.csv")

# %%
treated_ids = {27, 28}  # Queens Midtown Tunnel, Hugh L. Carey Tunnel
df_daily["group"] = df_daily["facility_id"].apply(lambda x: "treated" if x in treated_ids else "control")

panel = df_daily.groupby(["date", "group"])["total_traffic"].sum().reset_index()
panel["date"] = pd.to_datetime(panel["date"])

# %%
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

# %%
# DiD regression
panel_long = panel.copy()
panel_long["post"] = (panel_long["date"] >= "2025-01-05").astype(int)
panel_long["treated"] = (panel_long["group"] == "treated").astype(int)
panel_long["log_traffic"] = np.log(panel_long["total_traffic"])
panel_long["dow"] = panel_long["date"].dt.dayofweek
panel_long = panel_long[panel_long["date"] >= "2023-01-01"]

# %%
model = smf.ols(
    "log_traffic ~ treated + post + treated:post + C(dow)",
    data=panel_long
).fit(cov_type="cluster", cov_kwds={"groups": panel_long["date"].dt.to_period("M")})

print(model.summary())

# %%
panel_long["resid"] = model.resid
outliers = panel_long.reindex(panel_long["resid"].abs().sort_values(ascending=False).index)
print(outliers[["date", "group", "total_traffic", "resid"]].head(15))

# %% Exclude known anomalous days and refit
exclude_dates = pd.to_datetime([
    "2026-02-22", "2026-02-23",  # NYC travel ban, historic blizzard
    "2026-01-25", "2026-01-26",  # earlier Jan 2026 snowstorm
    "2023-07-04", "2024-07-04",  # July 4th
    "2024-12-25", "2025-12-27",  # Christmas / day after
])

panel_long_clean = panel_long[~panel_long["date"].isin(exclude_dates)]

model_clean = smf.ols(
    "log_traffic ~ treated + post + treated:post + C(dow)",
    data=panel_long_clean
).fit(cov_type="cluster", cov_kwds={"groups": panel_long_clean["date"].dt.to_period("M")})

print(model_clean.summary())
# %% Full diagnostics
print(model_clean.summary().as_text())
# %%
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(model_clean.resid)
print(f"Durbin-Watson: {dw_stat:.3f}")
# %%
resid_treated = panel_long_clean.loc[panel_long_clean["treated"] == 1, "resid"].sort_index()
resid_control = panel_long_clean.loc[panel_long_clean["treated"] == 0, "resid"].sort_index()

print(f"Treated DW: {durbin_watson(resid_treated):.3f}")
print(f"Control DW: {durbin_watson(resid_control):.3f}")
# %%
# pip install linearmodels --break-system-packages   (if not already installed)

panel_indexed = panel_long_clean.copy()
panel_indexed["entity"] = panel_indexed["group"]  # "treated" or "control"
panel_indexed = panel_indexed.set_index(["entity", "date"])
# %%
from linearmodels.panel import PanelOLS

panel_indexed["treated_post"] = panel_indexed["treated"] * panel_indexed["post"]

mod = PanelOLS.from_formula(
    "log_traffic ~ treated_post + EntityEffects + TimeEffects",
    data=panel_indexed
)
res = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=7)

print(res)
# %%
dw_final = durbin_watson(res.resids)
print(f"Post-fit DW: {dw_final:.3f}")
# %%
resid_series = res.resids.copy()
resid_series.index = panel_indexed.index  # restore entity/date multiindex

resid_treated_final = resid_series.xs("treated", level="entity").sort_index()
resid_control_final = resid_series.xs("control", level="entity").sort_index()

print(f"Treated DW: {durbin_watson(resid_treated_final):.3f}")
print(f"Control DW: {durbin_watson(resid_control_final):.3f}")
# %%
for bw in [4, 7, 14, 21]:
    r = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)
    print(f"bandwidth={bw}: coef={r.params['treated_post']:.4f}, se={r.std_errors['treated_post']:.4f}, p={r.pvalues['treated_post']:.4f}")
# %%
