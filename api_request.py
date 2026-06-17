import os
import mlflow
from fastapi import FastAPI
from fetch_data import (
    fetch_day_ahead,
    fetch_renewable_forecast,
    fetch_loads,
    inference_data_merger)
from features import feature_builder, prepare_inference_features
from contextlib import asynccontextmanager
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()
S3_url = os.getenv("S3_URL")

loaded_model = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global loaded_model
    loaded_model = mlflow.pyfunc.load_model(S3_url)
    yield
    loaded_model = None

app = FastAPI(lifespan=lifespan)

class ForecastRequest(BaseModel):
    start_time: str
    horizon: int

@app.post("/forecast")
def data(request: ForecastRequest):
    start_time = request.start_time
    horizon = request.horizon
    try:
        df_price_day_ahead = fetch_day_ahead(start_time=start_time,
                                         horizon=horizon,
                                         )
        df_forecast_renewables = fetch_renewable_forecast(start_time=start_time,
                                                      horizon=horizon)
        df_loads = fetch_loads(start_time=start_time,
                           horizon=horizon)
        merged, output_path_merged = inference_data_merger(
            df_price=df_price_day_ahead,
            df_load=df_loads,
            df_renewable_forecast=df_forecast_renewables)

        df, df_feature, gold_output_path = feature_builder(output_path_merged)

        X = prepare_inference_features(gold_output_path)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail= f"Failed to fetch/build data: {str(e)}"
        )

    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        preds = loaded_model.predict(X)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Failed to run model prediction: {str(e)}",
                "input_columns": X.columns.tolist(),
                "input_shape": list(X.shape),
            }
        )


    return {"start_date": start_time,
            "horizon": horizon,
            "predictions": preds.tolist(),
            "timestamps": df_feature.index.astype(str).tolist()}
