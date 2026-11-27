"""
Повторний експорт одного року, розбитий на частини: окреме завдання для кожної області.
Маленькі завдання не впираються в ліміт часу (помилка "Computation timed out").
Запуск:  python scripts/06_export_year_by_oblast.py 2025
Результат – 9 файлів ndvi_districts_2025_<область>.csv на Google Диску (папка farm-ai-thesis).
"""
import sys, pathlib, warnings
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import ee
from src import gee_data as g

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
OBLAST_PATTERNS = ["Poltav", "Khark", "Sums", "Cherkas", "Kirovoh",
                   "Dnipro", "Kyivs", "Chernih", "Vinnyt"]
SCALE = 30
WINDOW = 10
N_WINDOWS = 24

g.init()

all_ukr = ee.FeatureCollection(g.GAUL2).filter(ee.Filter.eq("GAUL0_NAME", "Ukraine"))
names = all_ukr.aggregate_array("GAUL1_NAME").distinct().sort().getInfo()
oblasts = [n for n in names if any(p.lower() in n.lower() for p in OBLAST_PATTERNS)]
print(f"Рік {YEAR}. Області:", ", ".join(oblasts))

crop = g.cropland_mask()
crop_count = crop.selfMask()
empty = ee.Image.constant(0).rename("NDVI").updateMask(0)
reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)


def oblast_table(dist, year):
    start = ee.Date(f"{year}-03-01")
    col = (g.s2_collection(dist, f"{year}-03-01", f"{year}-11-01")
           .map(lambda im: ee.Image(im).updateMask(crop)
                .copyProperties(im, ["system:time_start"])))
    starts = ee.List.sequence(0, N_WINDOWS - 1).map(
        lambda k: start.advance(ee.Number(k).multiply(WINDOW), "day").millis())

    def per_district(f):
        geom = f.geometry()
        total = ee.Number(crop_count.reduceRegion(ee.Reducer.count(), geom, SCALE,
                                                  maxPixels=1e10, tileScale=16).values().get(0))
        dcol = col.filterBounds(geom)

        def per_window(t):
            d0 = ee.Date(t)
            win = dcol.filterDate(d0, d0.advance(WINDOW, "day"))
            img = ee.Image(ee.Algorithms.If(win.size().gt(0), win.select("NDVI").median(), empty))
            st = img.reduceRegion(reducer, geom, SCALE, maxPixels=1e10, tileScale=16)
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


for ob in oblasts:
    dist = all_ukr.filter(ee.Filter.eq("GAUL1_NAME", ob))
    task = ee.batch.Export.table.toDrive(
        collection=oblast_table(dist, YEAR),
        description=f"ndvi_districts_{YEAR}_{ob}",
        folder="farm-ai-thesis",
        fileNamePrefix=f"ndvi_districts_{YEAR}_{ob}",
        fileFormat="CSV",
        selectors=["oblast", "district", "district_code", "year", "date",
                   "ndvi", "valid_share", "n_images"],
    )
    task.start()
    print(f"  {ob}: надіслано")

print("\nГотово! Ноутбук можна вимикати. Стан:  python scripts/05_task_status.py")