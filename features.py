import os
import pandas as pd
import numpy as np
import logging
import datetime as dt
from config import EXPECTED_COLUMNS_AND_ORDER


os.makedirs('logs', exist_ok=True)

def _logger(df: pd.DataFrame,
            file_name: str):
    """
    Logs summary statistics and metadata of the provided DataFrame to a log file. Generates a log file
    under the 'logs' directory with a timestamped filename and writes logging information
    such as missing values, shape of the DataFrame, column statistics, and key information.

    :param df: The DataFrame whose details and statistics will be logged.
    :type df: pandas.DataFrame
    :param file_name: The base name of the log file to be created, excluding directory and timestamp.
    :type file_name: str
    :return: None
    """

    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/{file_name}_{dt.datetime.now().strftime("%Y_%m_%d_%H_%M")}.log'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s  %(message)s'))
    logger.addHandler(fh)


    missing = df.isna().sum().sum()
    if missing > 0 :
        logger.warning(f"There are missing values ❌ : {df.isna().sum()}")
        logger.warning(f"Missing total: {missing}")
    else:
        logger.info(f"There are no missing values ✅ ")


    logger.info(f'Shape: {df.shape}')
    logger.info(f'Keys: {df.keys()}')
    logger.info("STD %s", df.std().to_dict())
    logger.info("Mean %s",  df.mean().to_dict())
    logger.info("Min/Max  %s / %s", df.min().to_dict(), df.max().to_dict())

    logger.removeHandler(fh)

def feature_builder(path: str,
                    index : int = 0,
                    lagged : int = 24,
                    tz  = "Europe/Brussels") -> tuple:

    """
    Builds and processes feature sets from a given CSV file containing time-series data. The function reads the data,
    applies transformations including time zone adjustments, feature engineering for cyclical time components,
    and lag variable creation. It saves the generated features to a CSV file and returns both the transformed complete
    dataframe and the dataframe of selected features.

    :param path: The file path of the input CSV file containing the raw data.
    :type path: str
    :param index: The starting index to process the data. Defaults to 0.
    :type index: int, optional
    :param lagged: Number of time steps for the lagged price feature. Defaults to 24.
    :type lagged: int, optional
    :param tz: Timezone used for converting the timestamp column. Defaults to "Europe/Brussels".
    :type tz: str, optional
    :return: A tuple containing the full processed dataframe and the dataframe with selected features.
    :rtype: tuple
    """


    df = pd.read_csv(path)

    ## Daylight saving
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    ## back to brussel
    df.index = df.index.tz_convert(tz)

    df.columns = df.columns.str.replace(" ","_")

    df['sine_h'] = np.sin(df.index.hour * 2 *np.pi/24)
    df['cos_h'] = np.cos(df.index.hour * 2 *np.pi/24)

    df['sine_month'] = np.sin(df.index.month * 2 *np.pi/12)
    df['cos_month'] = np.cos(df.index.month * 2 *np.pi/12)

    df['dow'] = df.index.dayofweek
    df['sine_dow'] = np.sin(df['dow'] * 2 *np.pi / 7)
    df['cos_dow']  = np.cos(df['dow'] * 2 *np.pi / 7)

    df['weekend'] = df['dow'].isin([5,6]).astype(int)

    price_mean = df["day_ahead_price"].mean()

    df[f'lagged_price_{lagged}'] = (
                                    df['day_ahead_price']
                                    .shift(lagged)
                                    .fillna(price_mean)
                                    )

    df['solar_sin_inter'] = df['Solar_forecast'] * df['sine_h']
    df['solar_cos_inter'] = df['Solar_forecast'] * df['cos_h']


    _logger(df=df, file_name='features_builder')

    os.makedirs('gold', exist_ok=True)

    columns_to_save = ['day_ahead_price','Forecasted_Load', 'Solar_forecast',
                       'Wind_Offshore_forecast', 'Wind_Onshore_forecast', 'sine_h', 'cos_h',
                       'sine_month', 'cos_month', 'dow', 'sine_dow', 'cos_dow', 'weekend',
                       'lagged_price_24', 'solar_sin_inter', 'solar_cos_inter']

    df_feature = df[columns_to_save].copy()
    df_feature['wind_forecast_total'] = df_feature['Wind_Onshore_forecast'] + df_feature['Wind_Offshore_forecast']
    gold_output_path = f'gold/features{path[-29:-4]}'

    df_feature.to_csv(gold_output_path)

    _logger(df=df_feature, file_name='features_saved')

    return df, df_feature, gold_output_path

def prepare_inference_features(path:str):

    df = pd.read_csv(path,
                     index_col=0,
                     parse_dates=True)



    if list(df.columns) == EXPECTED_COLUMNS_AND_ORDER:
        print("Columns match and order is correct")
    else:
        missing = [c for c in EXPECTED_COLUMNS_AND_ORDER if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLUMNS_AND_ORDER]
        raise KeyError(f"Column mismatch. Missing: {missing}. Extra: {extra}")

    X = df.drop(columns='day_ahead_price')

    return X


if __name__ == '__main__':
    path = 'fetched_data/merged/merged_2024-01-01_to_2024-01-05.csv'
    df, df_feature, gold_output_path = feature_builder(path)

    # cors = df_feature.corr()
    # corr_one = df_feature.corr(numeric_only=True)["day_ahead_price"].dropna().sort_values()
    # corr_one.plot(kind="barh", figsize=(8, 10))
    # plt.title("Correlation with day_ahead_price")
    # plt.tight_layout()
    # plt.yticks(rotation=45)
    # plt.show()


