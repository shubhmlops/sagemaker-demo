# inference.py
import json
import joblib
import pandas as pd
import numpy as np
import os
from data_processing import prepare_inference_history

history_df = prepare_inference_history(history_df)


FEATURES = [
    "price", "promo_flag", "month", "year",
    "lag_1", "lag_3", "lag_6", "lag_12",
    "rolling_3", "rolling_6"
]


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def forecast_next_month(history, model):
    row = history.iloc[-1:].copy()
    X = row[FEATURES]
    prediction = model.predict(X)[0]
    return prediction


def predict_fn(input_data, model):
    """
    input_data: historical dataframe (last 12+ months)
    """
    history = input_data.copy()
    forecasts = []

    for i in range(120):  # 10 years
        next_date = history["date"].max() + pd.DateOffset(months=1)

        next_row = {
            "date": next_date,
            "price": history.iloc[-1]["price"],
            "promo_flag": 0,
            "month": next_date.month,
            "year": next_date.year,
            "lag_1": history.iloc[-1]["units_sold"],
            "lag_3": history.iloc[-3]["units_sold"],
            "lag_6": history.iloc[-6]["units_sold"],
            "lag_12": history.iloc[-12]["units_sold"],
            "rolling_3": history["units_sold"].tail(3).mean(),
            "rolling_6": history["units_sold"].tail(6).mean(),
        }

        df_next = pd.DataFrame([next_row])
        y_pred = model.predict(df_next[FEATURES])[0]

        next_row["units_sold"] = y_pred
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        forecasts.append({"date": str(next_date.date()), "forecast": float(y_pred)})

    return forecasts
