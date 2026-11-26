"""
Розвідка бази меж районів FAO GAUL 2025:
  1) які поля має район, у якому лежить м. Хорол;
  2) які райони входять до тієї самої області;
  3) пробний ряд NDVI ріллі цього району за 2021 рік.
Запуск:  python scripts/02_check_gaul2025.py
"""
import sys, pathlib, warnings
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ee
from src import gee_data as g

g.init()
fc = ee.FeatureCollection(g.GAUL2)
khorol = ee.Geometry.Point([33.27, 49.78])

# 1) Район, у якому лежить Хорол, і всі його поля
feat = fc.filterBounds(khorol).first()
props = feat.toDictionary().getInfo()
print("1) Поля району, де лежить Хорол:")
for k, v in props.items():
    print(f"   {k} = {v}")

keys = list(props.keys())
k1 = next((k for k in keys if "1" in k and "name" in k.lower()), None)
k2 = next((k for k in keys if "2" in k and "name" in k.lower()), None)
print(f"\n   колонка області: {k1}, колонка району: {k2}")

# 2) Усі райони тієї ж області
if k1 and k2:
    same = fc.filter(ee.Filter.eq(k1, props[k1]))
    names = same.aggregate_array(k2).sort().getInfo()
    print(f"\n2) Райони області «{props[k1]}»: {len(names)}")
    print("   " + ", ".join(names))

# 3) Пробний ряд NDVI ріллі району, де лежить Хорол
print(f"\n3) NDVI ріллі, район «{props.get(k2)}», 2021 рік (1–3 хв)...")
df = g.ndvi_timeseries(feat.geometry(), "2021-03-01", "2021-11-01")
print(df.to_string(index=False) if not df.empty else "   порожньо")
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "test_ndvi_district_2021.csv"
df.to_csv(out, index=False)
print(f"\nЗбережено: {out}")