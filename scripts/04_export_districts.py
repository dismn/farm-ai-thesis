"""
Дорахунок років, які впали з помилкою (наприклад "Computation timed out").
Робить те саме, що 04_export_districts.py, ТИМ САМИМ методом,
але ділить рік на окремі завдання ПО ОБЛАСТЯХ — кожне завдання менше і встигає.
Результат – CSV-файли на Google Диску в папці farm-ai-thesis:
    ndvi_districts_2025_01_Cherkas_ka.csv, ndvi_districts_2025_02_... і т.д.
Запуск:  python scripts/04b_export_by_oblast.py
"""
import sys, pathlib, warnings, re
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ee
from src import gee_data as g

# ---------- НАЛАШТУВАННЯ ----------
YEARS = [2025]        # які роки дорахувати
ONLY_OBLASTS = []     # порожньо = всі області; або частини назв, напр. ["Khark", "Dnipro"]
SPLIT_SEASON = False  # True = ще й ділити сезон навпіл (якщо якась область знову впаде)
# ----------------------------------

# Ці значення ОБОВ'ЯЗКОВО такі самі, як у 04_export_districts.py
OBLAST_PATTERNS = ["Poltav", "Khark", "Sums", "Cherkas", "Kirovoh",
                   "Dnipro", "Kyivs", "Chernih", "Vinnyt"]
SCALE = 30
WINDOW = 10
N_WINDOWS = 24

g.init()

all_ukr = ee.FeatureCollection(g.GAUL2).filter(ee.Filter.eq("GAUL0_NAME", "Ukraine"))
names = all_ukr.aggregate_array("GAUL1_NAME").distinct().sort().getInfo()
oblasts = [n for n in names if any(p.lower() in n.lower() for p in OBLAST_PATTERNS)]
if ONLY_OBLASTS:
    oblasts = [n for n in oblasts if any(p.lower() in n.lower() for p in ONLY_OBLASTS)]
print("Області:", ", ".join(oblasts))

crop = g.cropland_mask()
crop_count = crop.selfMask()
empty = ee.Image.constant(0).rename("NDVI").updateMask(0)
reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)


def year_table(year, dist, k_from, k_to):
    """Те саме, що в 04_export_districts.py, але для заданих районів і вікон k_from..k_to-1."""
    start = ee.Date(f"{year}-03-01")
    col = (g.s2_collection(dist, f"{year}-03-01", f"{year}-11-01")
           .map(lambda im: ee.Image(im).updateMask(crop)
                .copyProperties(im, ["system:time_start"])))
    starts = ee.List.sequence(k_from, k_to - 1).map(
        lambda k: start.advance(ee.Number(k).multiply(WINDOW), "day").millis())

    def per_district(f):
        geom = f.geometry()
        total = ee.Number(crop_count.reduceRegion(ee.Reducer.count(), geom, SCALE,
                                                  maxPixels=1e10, tileScale=8).values().get(0))
        dcol = col.filterBounds(geom)

        def per_window(t):
            d0 = ee.Date(t)
            win = dcol.filterDate(d0, d0.advance(WINDOW, "day"))
            img = ee.Image(ee.Algorithms.If(win.size().gt(0), win.select("NDVI").median(), empty))
            st = img.reduceRegion(reducer, geom, SCALE, maxPixels=1e10, tileScale=8)
            cnt = ee.Number(ee.Algorithms.If(st.get("NDVI_count"), st.get("NDVI_count"), 0))
            return ee.Feature(None, {
                "oblast": f.get("GAUL1_NAME"),
                "district": f.get("GAUL2_NAME"),
                "district_code": f.get("GAUL2_CODE"),
                "year": year,
                "date": d0.advance(WINDOW / 2, "day").format("YYYY-MM-dd"),
                "ndvi": st.get("NDVI_mean"),
                "valid_share": cnt.divide(total),
                "n_images": win.size(),
            })
        return ee.FeatureCollection(starts.map(per_window))

    return dist.map(per_district).flatten()


def safe(text):
    """Earth Engine не любить апострофи й пробіли в назвах завдань."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


if SPLIT_SEASON:
    half = N_WINDOWS // 2
    parts = [(0, half, "a"), (half, N_WINDOWS, "b")]   # a = березень–червень, b = липень–жовтень
else:
    parts = [(0, N_WINDOWS, "")]

for year in YEARS:
    for i, oblast in enumerate(oblasts, start=1):
        dist = all_ukr.filter(ee.Filter.eq("GAUL1_NAME", oblast))
        for k_from, k_to, suffix in parts:
            name = f"ndvi_districts_{year}_{i:02d}_{safe(oblast)}" + (f"_{suffix}" if suffix else "")
            task = ee.batch.Export.table.toDrive(
                collection=year_table(year, dist, k_from, k_to),
                description=name[:100],
                folder="farm-ai-thesis",
                fileNamePrefix=name,
                fileFormat="CSV",
                selectors=["oblast", "district", "district_code", "year", "date",
                           "ndvi", "valid_share", "n_images"],
            )
            task.start()
            print(f"Надіслано: {name}")

print("\nГотово! Ноутбук можна вимикати.")
print("Стан завдань:  python scripts/05_task_status.py")
