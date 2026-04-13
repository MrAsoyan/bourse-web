import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import yfinance as yf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ========================= DATA CACHE =========================
_data_cache = {}


def date2angle(dates):
    if not isinstance(dates, pd.Series):
        dates = pd.to_datetime(dates)
    leap_year = (dates.year % 4 == 0) & (dates.year % 100 != 0) | (dates.year % 400 == 0)
    return np.where(leap_year, dates.dayofyear * 2 * np.pi / 366, dates.dayofyear * 2 * np.pi / 365)


def normalizer(s):
    serie = pd.Series(s)
    return (serie - serie.min()) / (serie.max() - serie.min())


LISTE_SYMBOLES = sorted([
    'NG=F', 'MNSO', 'PDD', 'BA', 'CUTR', 'VTLE', 'U', 'GPS', 'BG', 'AAL',
    'RH', 'ASO', 'TCEHY', 'LX', 'GOTU', '0700.HK', 'DAO', 'QFIN', '1810.HK',
    'TCOM', 'HUYA', 'BEKE', 'ATIP', 'KO', 'KD', 'PEP', 'MCD', 'ULTA', 'BBY',
    'BBWI', 'NVDA', 'ATO.PA', 'AMWL', 'BOXL', 'TAL', 'IRT', 'AF.PA', 'ZURA',
    'META', 'OCGN', 'AMC', 'ABEV', 'SONY', '6758.T', 'NAS.OL', 'GAP', 'EDU',
    'NPWR', 'EXC', 'FSRNQ', 'CECO', '006120.KS', 'YINN', 'TSLA', 'LRN',
    'AAPL', 'GOOG', 'MSFT', 'AMZN',
])


@app.route("/")
def index():
    return render_template("index.html",
                           symbols=LISTE_SYMBOLES,
                           current_year=int(time.strftime("%Y")))


@app.route("/api/data")
def api_data():
    symbol = request.args.get("symbol", "AAPL")
    start = request.args.get("start", "2018")

    key = (symbol, start)
    if key in _data_cache:
        return jsonify(_data_cache[key])

    start_str = f"{start}-01-01" if len(str(start)) == 4 else str(start)
    df = yf.download(symbol, start=start_str, end="2027-01-01", interval="1d", progress=False)
    if df.empty:
        return jsonify({"error": "No data found"}), 404

    df.columns = df.columns.get_level_values(0)
    price = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    volume = df['Volume']

    wl = min(30, len(df) - 1)
    if wl < 5:
        sg = price.values.tolist()
    else:
        if wl % 2 == 0:
            wl -= 1
        sg = savgol_filter(price, window_length=wl, polyorder=min(4, wl - 1)).tolist()

    result = {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "price": price.values.tolist(),
        "volume": volume.values.tolist(),
        "savgol": sg,
    }
    _data_cache[key] = result
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
