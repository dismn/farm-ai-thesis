"""
Крок 1 збирання даних: що є в Earth Engine для Полтавської області.
  1) які райони є в базі FAO GAUL;
  2) скільки знімків Sentinel-2 за сезон (квітень–вересень) у кожному році;
  3) пробний часовий ряд NDVI для одного району за один рік.
Запуск (з папки проєкту):  python scripts/01_check_availability.py
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))   # щоб бачити папку src

import ee
from src import gee_data as g

g.init()
print("Підключення: OK\n")

# 1) Райони
names = g.districts().aggregate_array("ADM2_NAME").sort().getInfo()
print(f"1) Районів Полтавської області в GAUL: {len(names)}")
print("   " + ", ".join(names) + "\n")

# 2) Кількість знімків по роках
oblast = g.districts().geometry()
print("2) Знімки Sentinel-2 над областю, квітень–вересень, хмарність < 80 %:")
for year in range(2017, 2026):
    n = (ee.ImageCollection(g.S2).filterBounds(oblast)
         .filterDate(f"{year}-04-01", f"{year}-10-01")
         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
         .size().getInfo())
    print(f"   {year}: {n} знімків")

# 3) Пробний ряд NDVI: перший район зі списку, 2021 рік, тільки рілля
first = g.districts().filter(ee.Filter.eq("ADM2_NAME", names[0])).geometry()
print(f"\n3) Часовий ряд NDVI ріллі, район «{names[0]}», 2021 рік (може тривати 1–2 хв)...")
df = g.ndvi_timeseries(first, "2021-03-01", "2021-11-01")
print(df.to_string(index=False))
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "test_ndvi_district_2021.csv"
df.to_csv(out, index=False)
print(f"\nЗбережено: {out}")
