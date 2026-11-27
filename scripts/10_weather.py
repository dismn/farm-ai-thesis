"""
Крок 10. Погода NASA POWER для всіх районів і агрометеорологічні ознаки сезонів.
  1) координати центрів районів беремо з Earth Engine (один маленький запит);
  2) для кожного району завантажуємо добову погоду з 1 вересня 2020 р. (51 запит, ~5–10 хв);
  3) рахуємо ознаки сезону для кожного року врожаю.
Результат:
  data/raw/district_centroids.csv, data/raw/weather_daily.csv,
  data/processed/weather_features.csv
Запуск:  python scripts/10_weather.py
(Якщо скрипт перервався, просто запустіть знову: вже завантажені райони він пропустить.)
"""
import sys, pathlib, time, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd
from src import weather as w

START, END = "20200901", "20251031"
YEARS = range(2021, 2026)
raw = ROOT / "data/raw"
cent_path, daily_path = raw / "district_centroids.csv", raw / "weather_daily.csv"

# 1) Центри районів (один раз)
if not cent_path.exists():
    import ee
    from src import gee_data as g
    g.init()
    codes = pd.read_csv(raw / "ndvi_districts_2021.csv")["district_code"].unique().tolist()
    fc = (ee.FeatureCollection(g.GAUL2).filter(ee.Filter.inList("GAUL2_CODE", codes))
          .map(lambda f: ee.Feature(None, {
              "district_code": f.get("GAUL2_CODE"), "district": f.get("GAUL2_NAME"),
              "oblast": f.get("GAUL1_NAME"),
              "lon": f.geometry().centroid(100).coordinates().get(0),
              "lat": f.geometry().centroid(100).coordinates().get(1)})))
    cent = pd.DataFrame([x["properties"] for x in fc.getInfo()["features"]])
    cent.to_csv(cent_path, index=False)
    print(f"Центри районів збережено: {len(cent)}")
cent = pd.read_csv(cent_path)

# 2) Добова погода (з продовженням після перерви)
done = pd.read_csv(daily_path, parse_dates=["date"]) if daily_path.exists() else pd.DataFrame()
done_codes = set(done["district_code"]) if len(done) else set()
parts = [done] if len(done) else []
todo = cent[~cent["district_code"].isin(done_codes)]
for i, r in enumerate(todo.itertuples(), 1):
    print(f"[{i}/{len(todo)}] {r.oblast}, {r.district}")
    df = w.daily_point(r.lat, r.lon, START, END)
    df.insert(0, "district_code", r.district_code)
    parts.append(df)
    pd.concat(parts, ignore_index=True).to_csv(daily_path, index=False)   # зберігаємо після кожного
    time.sleep(1)                                                         # не перевантажуємо сервер
daily = pd.concat(parts, ignore_index=True)
print(f"Добових записів: {len(daily)}, районів: {daily['district_code'].nunique()}")

# 3) Ознаки сезону
rows = []
for code, d in daily.groupby("district_code"):
    for y in YEARS:
        rows.append({"district_code": code, "year": y, **w.season_features(d, y)})
feat = pd.DataFrame(rows).merge(cent[["district_code", "district", "oblast"]], on="district_code")
(ROOT / "data/processed").mkdir(exist_ok=True)
feat.to_csv(ROOT / "data/processed/weather_features.csv", index=False)

print("\nЛубенський район:")
print(feat[feat["district"] == "Lubenskyi"].drop(columns=["district_code", "oblast"]).to_string(index=False))