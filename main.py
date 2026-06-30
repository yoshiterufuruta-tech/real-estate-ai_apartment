from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()

# static フォルダをマウント
app.mount("/static", StaticFiles(directory="static"), name="static")

STATIC_DIR = Path("static")

# モデル読み込み
model = joblib.load("model.pkl")

preprocess = model.named_steps["preprocess"]
regressor = model.named_steps["regressor"]

# 平均価格データ読み込み
with open(STATIC_DIR / "city_avg_price.json", encoding="utf-8") as f:
    city_avg_price = json.load(f)

with open(STATIC_DIR / "district_avg_price.json", encoding="utf-8") as f:
    district_avg_price = json.load(f)

feature_columns = preprocess.get_feature_names_out()

class PredictRequest(BaseModel):
    都道府県名: str
    市区町村名: str
    地区名: str
    面積: float
    間取り: str
    築年数: float
    駅距離: float
    用途: str
    構造: str

def normalize_floor_plan(x):
    if pd.isna(x): return "不明"
    s = str(x).upper()
    s = s.replace("Ｌ", "L").replace("Ｄ", "D").replace("Ｋ", "K").replace("Ｒ", "R")
    s = s.replace("Ｓ", "S").replace(" ", "").replace(" ", "")
    if "1R" in s or "1K" in s: return "1K"
    if "1DK" in s: return "1DK"
    if "1LDK" in s: return "1LDK"
    if "2K" in s: return "2K"
    if "2DK" in s: return "2DK"
    if "2LDK" in s: return "2LDK"
    if "3LDK" in s: return "3LDK"
    if "4LDK" in s: return "4LDK"
    if "5LDK" in s: return "5LDK"
    return "その他"
    
@app.post("/predict")
def predict(req: PredictRequest):
    logging.info("REQUEST JSON: %s", req.model_dump())
    city_avg = city_avg_price.get(req.市区町村名, 0)
    district_avg = district_avg_price.get(req.地区名, 0)

    raw = pd.DataFrame([{
        "都道府県名": req.都道府県名,
        "市区町村名": req.市区町村名,
        "地区名": req.地区名,
        "面積": req.面積,
        "築年数": req.築年数,
        "間取り": req.間取り,
        "駅距離": req.駅距離,
        "用途": req.用途,
        "構造": req.構造,
        "市区町村平均価格": city_avg,
        "地区平均価格": district_avg,
        "市区町村平均価格_log": np.log1p(city_avg),
        "地区平均価格_log": np.log1p(district_avg)
    }])

    raw["駅距離_log"] = np.log1p(raw["駅距離"])
    raw["面積_sqrt"] = np.sqrt(raw["面積"])

    X = preprocess.transform(raw)
    pred = regressor.predict(X)[0]

    pred = pred * (122.1 / 119.2)
    pred_list_price = pred * 1.255

    pred = max(pred, 0)

    return {
        "predicted_price": int(pred),
        "predicted_list_price": int(pred_list_price)
    }
