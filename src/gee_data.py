"""
М1. Збирання даних з Google Earth Engine.
Функції реалізують модель обробки часових рядів NDVI з підрозділу 2.2.
"""
import ee
import pandas as pd

PROJECT_ID = "crucial-garden-425418-h3"

S2 = "COPERNICUS/S2_SR_HARMONIZED"        # Sentinel-2, рівень 2A (Sen2Cor)
GAUL2 = "FAO/GAUL/2025/level2"            # межі районів (після реформи 2020 р.)
WORLDCOVER = "ESA/WorldCover/v200"        # карта землекористування, клас 40 = рілля

# Класи карти SCL, які залишаємо: 4 – рослинність, 5 – відкритий ґрунт (2.2, маскування хмар)
SCL_KEEP = [4, 5]


def init():
    """Підключення до Earth Engine (перший раз відкриє браузер для входу)."""
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)


def _get_with_retry(obj, tries=6):
    """getInfo() з повторами: якщо сервер перевантажений (429), чекаємо й пробуємо знову."""
    import time
    for k in range(tries):
        try:
            return obj.getInfo()
        except ee.EEException as e:
            if "Too many" in str(e) or "429" in str(e):
                wait = 5 * (k + 1)
                print(f"   сервер зайнятий, чекаю {wait} с...")
                time.sleep(wait)
            else:
                raise
    return obj.getInfo()


def cropland_mask():
    """Маска орних земель: 1 там, де рілля (ESA WorldCover, клас 40)."""
    wc = ee.ImageCollection(WORLDCOVER).first()
    return wc.eq(40)


def mask_and_ndvi(img):
    """Маскує хмари за SCL і додає канал NDVI = (B8 - B4) / (B8 + B4), формула (2.6)."""
    scl = img.select("SCL")
    good = scl.eq(SCL_KEEP[0])
    for c in SCL_KEEP[1:]:
        good = good.Or(scl.eq(c))
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ndvi.updateMask(good).copyProperties(img, ["system:time_start"])


def s2_collection(region, start, end, max_cloud=80):
    """Знімки Sentinel-2 над ділянкою за період, з NDVI і маскою хмар."""
    return (ee.ImageCollection(S2)
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .map(mask_and_ndvi))


def ndvi_timeseries(region, start, end, scale=20, only_cropland=True, min_valid=0.8,
                    composite_days=None):
    """
    Часовий ряд середнього NDVI ділянки, формула (2.7).
    Повертає таблицю pandas: date, ndvi, valid_share.

    composite_days=None – щоденний ряд (для полів: поле потрапляє в один знімок);
    composite_days=10   – 10-денні композити (для районів: район знімається частинами
                          в різні дні, тому знімки за 10 днів «склеюються» медіаною).
    Спостереження з часткою допустимих пікселів < min_valid відкидаються (поріг η, 2.2).
    """
    col = s2_collection(region, start, end)
    if only_cropland:
        crop = cropland_mask()
        col = col.map(lambda im: ee.Image(im).updateMask(crop)
                      .copyProperties(im, ["system:time_start"]))
        total_mask = crop.selfMask()
    else:
        total_mask = ee.Image.constant(1)

    total = ee.Number(total_mask.reduceRegion(ee.Reducer.count(), region, scale,
                                              maxPixels=1e10, tileScale=4).values().get(0))
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    def stats_feature(img, date):
        st = ee.Image(img).reduceRegion(reducer, region, scale, maxPixels=1e10, tileScale=4)
        return ee.Feature(None, {
            "date": ee.Date(date).format("YYYY-MM-dd"),
            "ndvi": st.get("NDVI_mean"),
            "valid_share": ee.Number(ee.Algorithms.If(st.get("NDVI_count"),
                                                      st.get("NDVI_count"), 0)).divide(total),
        })

    if composite_days:
        # Десятиденки рахуємо ПО ЧЕРЗІ: безкоштовний тариф Earth Engine
        # не дозволяє багато важких обчислень одночасно (помилка 429).
        empty = ee.Image.constant(0).rename("NDVI").updateMask(0)
        rows = []
        t0 = pd.Timestamp(start)
        t_end = pd.Timestamp(end)
        while t0 + pd.Timedelta(days=composite_days) <= t_end:
            t1 = t0 + pd.Timedelta(days=composite_days)
            win = col.filterDate(t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"))
            img = ee.Image(ee.Algorithms.If(win.size().gt(0), win.select("NDVI").median(), empty))
            mid = t0 + pd.Timedelta(days=composite_days / 2)
            props = _get_with_retry(stats_feature(img, mid.strftime("%Y-%m-%d")).toDictionary())
            if props.get("ndvi") is not None:
                rows.append(props)
            t0 = t1
        feats = [{"properties": r} for r in rows]
    else:
        fc = col.map(lambda im: stats_feature(im, im.get("system:time_start")))
        feats = _get_with_retry(fc.filter(ee.Filter.notNull(["ndvi"])))["features"]

    df = pd.DataFrame([f["properties"] for f in feats])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    if not composite_days:
        # кілька «плиток» за одну дату – усереднюємо, частки складаємо
        df = df.groupby("date", as_index=False).agg(ndvi=("ndvi", "mean"),
                                                    valid_share=("valid_share", "sum"))
    df["valid_share"] = df["valid_share"].clip(upper=1).round(3)
    df["ndvi"] = df["ndvi"].round(4)
    return df[df["valid_share"] >= min_valid].sort_values("date").reset_index(drop=True)


def district_by_name(name, oblast="Poltavska"):
    """Геометрія району за назвою з GAUL 2025 (напр. 'Lubenskyi')."""
    return (ee.FeatureCollection(GAUL2)
            .filter(ee.Filter.eq("GAUL1_NAME", oblast))
            .filter(ee.Filter.eq("GAUL2_NAME", name))
            .geometry())


def districts(oblast="Poltavska"):
    """Райони області з бази FAO GAUL 2025 (межі після реформи 2020 року)."""
    return (ee.FeatureCollection(GAUL2)
            .filter(ee.Filter.eq("GAUL0_NAME", "Ukraine"))
            .filter(ee.Filter.eq("GAUL1_NAME", oblast)))