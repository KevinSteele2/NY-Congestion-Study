import requests
import pandas as pd

url = "https://data.ny.gov/resource/ebfx-2m7v.json"
resp = requests.get(url, params={
    "$select": "date, facility_id, facility, sum(traffic_count) as total_traffic",
    "$group": "date, facility_id, facility",
    "$limit": 50000
})
df_daily = pd.DataFrame(resp.json())
df_daily["total_traffic"] = df_daily["total_traffic"].astype(int)
df_daily["facility_id"] = df_daily["facility_id"].astype(int)

df_daily.to_csv("daily_facility_traffic.csv", index=False)
print(f"Saved {len(df_daily)} rows")