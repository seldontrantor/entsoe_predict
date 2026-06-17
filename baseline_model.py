import os
import logging
import datetime as dt
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature

os.makedirs('logs', exist_ok=True)
log_file = f'logs/baseline_model_{dt.datetime.now().strftime("%Y-%m-%d_%H_%M")}.log'
logging.basicConfig(filename=log_file,
                    format='%(asctime)s  %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)

def train_test_split(path: str,
                     train_start: str,
                     train_end: str,
                     target_col: str = "day_ahead_price",
                     index_col: int = 0,
                     parse_dates: bool = True) -> tuple:

    df = pd.read_csv(path,
                     index_col=index_col,
                     parse_dates=parse_dates)

    tz  = "Europe/Brussels"

    train_start = pd.Timestamp(train_start, tz=tz)
    train_end = pd.Timestamp(train_end, tz=tz)
    test_start = train_end + pd.Timedelta(15, unit='min')

    train = df.loc[train_start:train_end]
    test = df.loc[test_start:]

    df_train_x = train.drop(columns=target_col).astype(float)
    df_train_y = train[target_col]

    df_test_x = test.drop(columns=target_col).astype(float)
    df_test_y = test[target_col]

    logger.info(f'Negative prices in train set: {sum(df_train_y<0)}')
    logger.info(f'Negative prices in test set: {sum(df_test_y<0)}')
    logger.info('Min/Max Price in train set %s / %s', df_train_y.min(), df_train_y.max())
    logger.info('Mean price in train set is %s ', df_train_y.mean())
    logger.info('Mean price in test set is %s ', df_test_y.mean())


    return df_train_x, df_train_y, df_test_x, df_test_y



def make_pipe(model,
              X_train,
              y_train,
              X_test,
              y_test):


    pipe = Pipeline(
                    [("model", model)]
                    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    MAE = mean_absolute_error(y_test,preds)
    RMSE = root_mean_squared_error(y_test,preds)
    R2 = r2_score(y_test,preds)

    log_file = f'logs/baseline_{dt.datetime.now().strftime("%Y_%m_%d_%H_%M")}.log'
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s  %(message)s',
                        level="INFO")
    logger = logging.getLogger(__name__)

    logger.info('MAE: %.2f', MAE)
    logger.info('RMSE: %.2f', RMSE)
    logger.info('R2: %.2f', R2)

    print(f'Model MAE: {MAE:.2f}')
    print(f'Model RMSE: {RMSE:.2f}')
    print(f'Model R2: {R2:.2f}')

    return preds, pipe


if __name__ == '__main__':
    path = 'gold/features_2024-01-01_to_2025-12-30'

    train_start="2024-01-01"
    train_end="2025-10-01"

    df_train_x, df_train_y, df_test_x, df_test_y = train_test_split(path,
                                                                    train_start=train_start,
                                                                    train_end=train_end,
                                                                    target_col="day_ahead_price")


    model = xgb.XGBRFRegressor(n_estimators=1000)
    preds, fitted_pipe = make_pipe(model, df_train_x, df_train_y, df_test_x, df_test_y )

    fitted_model = fitted_pipe.named_steps["model"]
    mlflow.set_experiment('day_ahead_price_forecast_DE_LU')
    with mlflow.start_run() as run:

        # log params
        mlflow.log_param('model_class', model.__class__.__name__)
        mlflow.set_tag("stage", "Staging")

        FI = fitted_model.feature_importances_
        feature_names = df_train_x.columns
        for name, score in zip(feature_names, FI):
            print(name, score)

        fig, ax = plt.subplots(dpi =200, tight_layout=True)
        ax.plot(preds, label='predicts')
        ax.plot(df_test_y.values, label='true')
        plt.legend()
        plt.show()

        # model metrics
        MAE = mean_absolute_error(df_test_y,preds)
        RMSE = root_mean_squared_error(df_test_y,preds)
        R2 = r2_score(df_test_y,preds)

        print(f"Xgb MAE: {MAE:.2f}")
        print(f"Xgb RMSE: {RMSE:.2f}")
        print(f"Xgb R2: {R2:.2f}")

        # log metrics for xgb
        mlflow.log_metric('MAE', float(MAE))
        mlflow.log_metric('RMSE', float(RMSE))
        mlflow.log_metric('R2', float(R2))

        # log params for xgb model:
        mlflow.log_param("train_start", train_start)
        mlflow.log_param("train_end", train_end)
        mlflow.log_param("clip_threshold", -150)
        mlflow.log_param("train_size", len(df_train_x))
        mlflow.log_param("test_size", len(df_test_x))

        # log input and signature
        input_example = df_train_x.head(5)
        signature = infer_signature(input_example, fitted_model.predict(input_example))

        # log model artifact (
        mlflow.xgboost.log_model(
            fitted_model,  # fitted model
            "model",
            model_format="json",
            signature=signature,
            input_example=input_example,
            registered_model_name="xgboost"
        )

        ## naive baseline:
        naive_mae = mean_absolute_error(df_test_y, np.full_like(df_test_y, df_test_y.mean()))
        logger.info('naive_mae: %.2f', naive_mae)
        print(f'naive_mae: {naive_mae}')

        lagged_baseline = (
            pd.concat([df_train_y, df_test_y])
            .shift(24)
            .loc[df_test_y.index]
        )
        valid_lagged = lagged_baseline.notna()
        y_true_lagged = df_test_y.loc[valid_lagged]
        y_pred_lagged = lagged_baseline.loc[valid_lagged]

        mae_lagged = mean_absolute_error(y_true_lagged, y_pred_lagged)
        rmse_lagged = root_mean_squared_error(y_true_lagged, y_pred_lagged)
        r2_lagged = r2_score(y_true_lagged, y_pred_lagged)

        print(f"Lagged baseline MAE: {mae_lagged:.2f}")
        print(f"Lagged baseline RMSE: {rmse_lagged:.2f}")
        print(f"Lagged baseline R2: {r2_lagged:.2f}")

        mlflow.log_metric("Lagged baseline MAE", float(mae_lagged))
        mlflow.log_metric("Lagged baseline RMSE", float(rmse_lagged))
        mlflow.log_metric("Lagged baseline R2", float(r2_lagged))
        mlflow.log_metric("naive_mae", float(naive_mae))

        logger.info(f"Lagged baseline MAE: {mae_lagged:.2f}")
        logger.info(f"Lagged baseline RMSE: {rmse_lagged:.2f}")
        logger.info(f"Lagged baseline R2: {r2_lagged:.2f}")
