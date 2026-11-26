"""
Перевірка доступу до Google Earth Engine.
Що робить: знаходить знімки Sentinel-2 біля м. Хорол за літо 2024 року
і рахує середній NDVI у квадраті 1 x 1 км.
Запуск:  python test_gee.py
"""
import ee

PROJECT_ID = "crucial-garden-425418-h3"   # проєкт Earth Engine

# 1. Вхід. Першого разу відкриється браузер: увійдіть у той самий Google-акаунт
#    і натисніть "Allow". Далі вхід запам'ятовується.
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
print("1) Підключення до Earth Engine: OK")

# 2. Ділянка: квадрат ~1 x 1 км біля м. Хорол (тестова, не поле фермера)
point = ee.Geometry.Point([33.27, 49.78])     # довгота, широта
area = point.buffer(500).bounds()

# 3. Знімки Sentinel-2 рівня 2A за червень-липень 2024, хмарність < 20 %
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(area)
      .filterDate("2024-06-01", "2024-07-31")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
n = s2.size().getInfo()
print(f"2) Знайдено знімків Sentinel-2: {n}")

# 4. NDVI = (B8 - B4) / (B8 + B4)  — формула (2.6) з розділу 2
def add_ndvi(img):
    return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))

if n > 0:
    first = add_ndvi(s2.sort("system:time_start").first())
    date = ee.Date(first.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    ndvi = first.select("NDVI").reduceRegion(
        reducer=ee.Reducer.mean(), geometry=area, scale=10).get("NDVI").getInfo()
    print(f"3) Перший знімок: {date}, середній NDVI = {ndvi:.3f}")
    print("\nВсе працює! Значення NDVI від 0,2 до 0,9 означає, що на ділянці є рослинність.")
else:
    print("Знімків не знайдено — напишіть Claude, розберемося.")
