import requests
import pandas as pd

url = "https://data.ny.gov/resource/t6yz-b64h.json"

url2 = "https://data.ny.gov/resource/ebfx-2m7v.json"
resp = requests.get(url2, params={
    "$select": "date, facility_id, facility, sum(traffic_count) as total_traffic",
    "$group": "date, facility_id, facility",
    "$limit": 50000
})
df_daily = pd.DataFrame(resp.json())
df_daily["total_traffic"] = df_daily["total_traffic"].astype(int)
print(df_daily.shape)
print(df_daily["date"].min(), df_daily["date"].max())