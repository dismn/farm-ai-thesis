"""
Пробний часовий ряд NDVI ріллі Лубенського району за 2021 рік:
  а) щоденні спостереження – скільки району видно в кожен день (діагностика);
  б) 10-денні композити – ряд, який підемо використовувати для районів.
Запуск:  python scripts/03_test_district_ndvi.py
"""
import sys, pathlib, warnings
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src import gee_data as g

g.init()
region = g.district_by_name("Lubenskyi")
root = pathlib.Path(__file__).resolve().parents[1]

print("а) Щоденні знімки (без фільтра): яку частку ріллі району видно (1–3 хв)...")
daily = g.ndvi_timeseries(region, "2021-03-01", "2021-11-01", min_valid=0)
print(f"   щоденних спостережень: {len(daily)}, максимальна видимість: {daily.valid_share.max():.0%}" if not daily.empty else "   порожньо")

print("\nб) 10-денні композити, поріг видимості 50 % (рахуються по черзі, 2–5 хв)...")
comp = g.ndvi_timeseries(region, "2021-03-01", "2021-11-01", min_valid=0.5, composite_days=10)
print(comp.to_string(index=False) if not comp.empty else "   порожньо")

comp.to_csv(root / "data" / "raw" / "ndvi_Lubenskyi_2021.csv", index=False)
print("\nЗбережено: data/raw/ndvi_Lubenskyi_2021.csv")