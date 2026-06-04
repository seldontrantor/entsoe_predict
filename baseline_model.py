import os
import logging
import datetime as dt
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


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

    df_train_x = train.drop(columns=target_col)
    df_train_y = train[target_col].values

    df_test_x = test.drop(columns=target_col)
    df_test_y = test[target_col].values

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

    print(f'MAE for baseline: {MAE:.2f}')
    print(f'RMSE for baseline: {RMSE:.2f}')
    print(f'R2 for baseline: {R2:.2f}')

    return preds, model


if __name__ == '__main__':
    path = 'gold/features_2026-01-01_to_2026-05-31'
    df_train_x, df_train_y, df_test_x, df_test_y = train_test_split(path,
                                                                    train_start="2026-01-01",
                                                                    train_end="2026-05-04",
                                                                    target_col="day_ahead_price")
    model = xgb.XGBRFRegressor(n_estimators=1000)
    preds, model = make_pipe(model, df_train_x, df_train_y, df_test_x, df_test_y )

    FI = model.feature_importances_
    feature_names = df_train_x.columns
    for name, score in zip(feature_names, FI):
        print(name, score)

    fig, ax = plt.subplots(dpi =200, tight_layout=True)
    ax.plot(preds, label='predicts')
    ax.plot(df_test_y, label='true')
    plt.legend()
    plt.show()

    ## naive baseline:
    naive_mae = mean_absolute_error(df_test_y, np.full_like(df_test_y, df_test_y.mean()))
    logger.info('naive_mae: %.2f', naive_mae)
    print(f'naive_mae: {naive_mae}')

    y_true = df_test_y
    y_pred = df_test_x["lagged_price_24"]

    mae_lagged = mean_absolute_error(y_true, y_pred)
    rmse_lagged = root_mean_squared_error(y_true, y_pred)
    r2_lagged = r2_score(y_true, y_pred)

    print(f"Baseline MAE: {mae_lagged:.2f}")
    print(f"Baseline RMSE: {rmse_lagged:.2f}")
    print(f"Baseline R2: {r2_lagged:.2f}")

    logger.info(f"Baseline MAE: {mae_lagged:.2f}")
    logger.info(f"Baseline RMSE: {rmse_lagged:.2f}")
    logger.info(f"Baseline R2: {r2_lagged:.2f}")