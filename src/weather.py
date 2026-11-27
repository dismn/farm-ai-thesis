"""
М1. Погодні дані NASA POWER (агрометеорологічні ознаки сезону, підрозділ 2.5).
NASA POWER – безкоштовний сервіс NASA з добовими даними про погоду на сітці 0,5° × 0,5°
(температура й опади – з реаналізу MERRA-2). Ключ доступу не потрібен.
"""
import time
import requests
import pandas as pd

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
# T2M – середня температура на висоті 2 м, °C; T2M_MAX / T2M_MIN – максимум / мінімум, °C;
# PRECTOTCORR – опади (скориговані), мм/добу; ALLSKY_SFC_SW_DWN – сонячна радіація, МДж/м²/добу
PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
FILL = -999


def daily_point(lat, lon, start, end, tries=4):
    """Добові дані для однієї точки. start, end – рядки 'YYYYMMDD'."""
    query = {"parameters": ",".join(PARAMS), "community": "AG", "latitude": round(lat, 4),
             "longitude": round(lon, 4), "start": start, "end": end, "format": "JSON"}
    for k in range(tries):
        try:
            r = requests.get(API, params=query, timeout=120)
            r.raise_for_status()
            par = r.json()["properties"]["parameter"]
            df = pd.DataFrame(par)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            return df.where(df > FILL + 1).rename_axis("date").reset_index()
        except (requests.RequestException, KeyError, ValueError) as e:
            wait = 10 * (k + 1)
            print(f"   помилка ({e.__class__.__name__}), повтор через {wait} с")
            time.sleep(wait)
    raise RuntimeError(f"Не вдалося отримати дані для точки {lat}, {lon}")


def season_features(daily, year):
    """Агрометеорологічні ознаки одного сезону (рік урожаю = year)."""
    d = daily.set_index("date")
    def win(a, b):
        return d.loc[f"{a}":f"{b}"]
    autumn_winter = win(f"{year - 1}-09-01", f"{year}-03-31")   # запаси вологи під озимі
    spring = win(f"{year}-04-01", f"{year}-06-30")              # кущіння, вихід у трубку, налив озимих
    summer = win(f"{year}-07-01", f"{year}-08-31")              # критичний період кукурудзи й соняшнику
    jja = win(f"{year}-06-01", f"{year}-08-31")
    season = win(f"{year}-04-01", f"{year}-09-30")
    gdd = (season["T2M"] - 10).clip(lower=0).sum()              # сума ефективних температур > 10 °C
    return {
        "prec_sep_mar": round(autumn_winter["PRECTOTCORR"].sum(), 1),
        "prec_apr_jun": round(spring["PRECTOTCORR"].sum(), 1),
        "prec_jul_aug": round(summer["PRECTOTCORR"].sum(), 1),
        "t_apr_jun": round(spring["T2M"].mean(), 2),
        "t_jul_aug": round(summer["T2M"].mean(), 2),
        "hot_days_jja": int((jja["T2M_MAX"] > 30).sum()),      # кількість днів зі спекою > 30 °C
        "gdd10_apr_sep": round(gdd, 0),
        "rad_apr_aug": round(win(f"{year}-04-01", f"{year}-08-31")["ALLSKY_SFC_SW_DWN"].sum(), 0),
    }