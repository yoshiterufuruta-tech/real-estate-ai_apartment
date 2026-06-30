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
    
@app.post("/predict")
def predict(req: PredictRequest):
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

    # Pipeline の前処理部分を正しく呼び出す
    X_np = model[:-1].transform(raw)

    # 前処理後の正しい列名を取得
    feature_names = model[:-1].get_feature_names_out()

    # numpy → DataFrame に戻す
    X = pd.DataFrame(X_np, columns=feature_names)

    # 学習時と同じ列順に揃える
    X = X[model.feature_names_in_]

    # 予測
    pred = model.predict(X)[0]

    # 補正
    pred = pred * (122.1 / 119.2)
    pred_list_price = pred * 1.255

    pred = max(pred, 0)

    return {
        "predicted_price": int(pred),
        "predicted_list_price": int(pred_list_price)

    print("MODEL STEPS:", model.named_steps)
    print("TYPE OF PREPROCESS:", type(model.named_steps["preprocess"]))

    }
