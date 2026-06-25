import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

panel_long["resid"] = model.resid
outliers = panel_long.reindex(panel_long["resid"].abs().sort_values(ascending=False).index)
print(outliers[["date", "group", "total_traffic", "resid"]].head(15))