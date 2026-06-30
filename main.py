from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# モデル（Pipeline: preprocess + regressor）
model = joblib.load("model.pkl")

# 平均価格データ
with open(STATIC_DIR / "city_avg_price.json", encoding="utf-8") as f:
    city_avg_price = json.load(f)

with open(STATIC_DIR / "district_avg_price.json", encoding="utf-8") as f:
    district_avg_price = json.load(f)

# ============================
#  間取り → 数値化（部屋数 + LDK + DK + K + S）
# ============================

def encode_madori(m):
    if not isinstance(m, str):
        return 0, 0, 0, 0, 0

    s = m.upper()

    # 部屋数
    nums = re.findall(r'\d+', s)
    rooms = int(nums[0]) if nums else 0

    # フラグ
    is_ldk = 1 if "LDK" in s else 0
    is_dk  = 1 if "DK" in s and "LDK" not in s else 0
    is_k   = 1 if "K" in s and "DK" not in s and "LDK" not in s else 0
    is_s   = 1 if "S" in s else 0

    return rooms, is_ldk, is_dk, is_k, is_s

# ============================
#  FastAPI 入力
# ============================

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

# ============================
#  推定API
# ============================

@app.post("/predict")
def predict(req: PredictRequest):

    city_avg = city_avg_price.get(req.市区町村名, 0)
    district_avg = district_avg_price.get(req.地区名, 0)

    # 入力データを DataFrame 化
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

    # 数値特徴量
    raw["駅距離_log"] = np.log1p(raw["駅距離"])
    raw["面積_sqrt"] = np.sqrt(raw["面積"])

    # ★ 間取り数値化（学習コードと完全同期）
    raw["部屋数"], raw["LDKフラグ"], raw["DKフラグ"], raw["Kフラグ"], raw["Sフラグ"] = \
        zip(*raw["間取り"].apply(encode_madori))

    # ★ Pipeline に raw をそのまま渡す（前処理は model が全部やる）
    pred = model.predict(raw)[0]

    # 補正
    pred = pred * (122.1 / 119.2)
    pred_list_price = pred * 1.255

    return {
        "predicted_price": int(max(pred, 0)),
        "predicted_list_price": int(pred_list_price)
    }
