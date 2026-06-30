from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

app = FastAPI()

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

# ★ 学習時の列名と完全一致させる（日本語）
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

@app.post("/predict")
def predict(req: PredictRequest):

    # 平均価格特徴量
    city_avg = city_avg_price.get(req.市区町村名, 0)
    district_avg = district_avg_price.get(req.地区名, 0)

    # ★ 学習時と完全に同じ列名で DataFrame を作る
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

    # ★ 学習時と同じ派生特徴量
    raw["駅距離_log"] = np.log1p(raw["駅距離"])
    raw["面積_sqrt"] = np.sqrt(raw["面積"])

    # 前処理 → 推定
    X = preprocess.transform(raw)
    pred = regressor.predict(X)[0]

    # 補正係数
    pred = pred * (122.1 / 119.2)
    pred_list_price = pred * 1.255

    pred = max(pred, 0)

    return {
        "predicted_price": int(pred),
        "predicted_list_price": int(pred_list_price)
    }
