import os
from dotenv import load_dotenv
import logging
from typing import Optional
import pandas as pd
import datetime as dt
from entsoe import EntsoePandasClient

load_dotenv()

API_KEY = os.getenv("ENTENSO_API")

os.makedirs('logs', exist_ok=True)


def _expected_15min_index(start_time: pd.Timestamp,
                          end_time: pd.Timestamp) -> pd.DatetimeIndex:
    expected_index = pd.date_range(start=start_time,
                                   end=end_time,
                                   freq='15min',
                                   inclusive='left')
    expected_index.name = 'time'
    return expected_index


def _safe_timestamp_for_filename(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d_%H-%M-%S")


def _align_to_expected_index(df: pd.DataFrame,
                             start_time: pd.Timestamp,
                             end_time: pd.Timestamp,
                             name: str) -> pd.DataFrame:
    expected_index = _expected_15min_index(start_time, end_time)
    missing_index = expected_index.difference(df.index)
    extra_index = df.index.difference(expected_index)

    logger = logging.getLogger(__name__)
    if len(missing_index) > 0:
        logger.warning("%s is missing %s expected timestamps", name, len(missing_index))
        logger.warning("First missing timestamps for %s: %s", name, missing_index[:5].tolist())
    if len(extra_index) > 0:
        logger.warning("%s has %s unexpected timestamps", name, len(extra_index))
        logger.warning("First unexpected timestamps for %s: %s", name, extra_index[:5].tolist())

    return df.reindex(expected_index)


def _interpolate_small_gaps(df: pd.DataFrame,
                            name: str,
                            limit: int = 4) -> pd.DataFrame:
    missing_before = int(df.isna().sum().sum())
    if missing_before == 0:
        return df

    filled = df.interpolate(method='time',
                            limit=limit,
                            limit_direction='both')
    missing_after = int(filled.isna().sum().sum())

    logger = logging.getLogger(__name__)
    logger.warning("%s missing values before interpolation: %s", name, missing_before)
    logger.warning("%s missing values after interpolation: %s", name, missing_after)

    return filled


def _assert_same_index(left: pd.DataFrame,
                       right: pd.DataFrame,
                       left_name: str,
                       right_name: str) -> None:
    if left.index.equals(right.index):
        return

    left_only = left.index.difference(right.index)
    right_only = right.index.difference(left.index)
    raise ValueError(
        f"{left_name} and {right_name} indexes do not match. "
        f"{left_name}-only timestamps: {left_only[:5].tolist()}; "
        f"{right_name}-only timestamps: {right_only[:5].tolist()}; "
        f"{left_name} length: {len(left)}, {right_name} length: {len(right)}"
    )


def _log_helper_df_stats(df,
                         name: str,
                         start_time: Optional[pd.Timestamp] = None,
                         end_time: Optional[pd.Timestamp] = None,
                         plant_type: Optional[str] = None,
                         logger_file: Optional[str] = None,
                         logger: Optional[logging.Logger] = None,
                         timegap: Optional = None
                         ):

    if logger is None:
        logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger_file:
        log_file = f'logs/{logger_file}_{dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s  %(message)s'))
        logger.addHandler(fh)


    missing = df.isna().sum().sum()


    if missing > 0 :
        logger.warning(f"There Are missing values ❌ : {df.isna().sum()}")
    else:
        logger.info(f"There are no missing values ✅ ")

    logger.info(f'Shape for {name}: {df.shape}')
    logger.info(f'Keys for {name}: {df.keys()}')
    logger.info("STD for %s: %s", name, df.std().to_dict())
    logger.info("Mean for %s: %s", name, df.mean().to_dict())
    logger.info("Min/Max for %s: %s / %s", name, df.min().to_dict(), df.max().to_dict())
    logger.info(f'Length of {name}: {len(df)}')

    if plant_type:
        logger.info(f"Type of plant: {plant_type}")
    if start_time and end_time:
        length = int(abs(end_time - start_time) / pd.Timedelta('15min'))
        logger.info(f"Fetched over: {start_time} and {end_time}")
        logger.info(f'Length of input per days {length} and the data len {len(df)}')

        if timegap != 15:
            logger.warning(f" The time gap between data is manually adjusted to be 15 mins instead of {timegap} ❌")
        else:
            logger.info(f" The time gap between data is 15 mins ✅")

        if length != len(df):
            logger.warning(
                f"Length of input per days {length} and the data len {len(df)} do not match\n"
                "Fetched data does not cover every expected 15-minute timestamp."
            )

    if logger_file:
        logger.removeHandler(fh)
        fh.close()


def fetch_gen_data (start_time:str,
                horizon:int =1,
                country_code:str='DE_LU',
                plant_type:str = "B16",
                client_type = EntsoePandasClient
                ) -> pd.DataFrame:

    """
    Fetches energy generation data for a specific plant type, country, and time range with 15 min granularity.

    The function queries energy generation data using the specified parameters, processes
    the input date and time range, and logs relevant information about fetched data, such
    as missing values, overall shape, and keys of the resulting dataset.

    :param start_time: The starting time for the data query in string format.
    :param horizon: The horizon in days from the starting time to calculate the time range.
        Defaults to 7 days.
    :param country_code: The country code representing the target country. Defaults to 'DE_LU'.
    :param plant_type: The type of the energy generation plant to filter the data. Defaults to "B16".
        options: B16: solar, B18: Wind off shore, B19: wind onshore,
    :return: A pandas DataFrame containing the queried energy generation data.
    :rtype: pandas.DataFrame
    """

    tz = "Europe/Brussels"
    start_time = pd.Timestamp(start_time, tz=tz)
    end_time = start_time+ pd.Timedelta(horizon, "D")

    client = client_type(api_key=API_KEY)
    data = client.query_generation(country_code=country_code,
                                   start=start_time,
                                   end=end_time,
                                   psr_type=plant_type,
                                   )
    data.index.name = 'time'
    data = _align_to_expected_index(data,
                                    start_time=start_time,
                                    end_time=end_time,
                                    name=f"generation {plant_type}")
    _log_helper_df_stats(data,
                         name="generation",
                         start_time=start_time,
                         end_time=end_time,
                         plant_type=plant_type,
                         logger_file="fetch_gen"
                         )

    return data

def concat_gen_data(start_time:str,
                    horizon:int =1,
                    country_code:str='DE_LU',
                    client_type = EntsoePandasClient,
                    assets :list = ['B16', 'B18','B19']
                    ) ->pd.DataFrame:

    data_each = []
    for i in assets:
        d = fetch_gen_data(start_time=start_time,
                       horizon=horizon,
                       country_code = country_code,
                       client_type=client_type,
                       plant_type=i
                       )
        if isinstance(d.columns, pd.MultiIndex):
            d = d.xs("Actual Aggregated", level=1, axis=1)
        data_each.append(d)

    data = pd.concat(data_each, axis=1)
    data.columns = ['Solar_gen', 'Wind_Offshore_gen', 'Wind_Onshore_gen']
    data = _interpolate_small_gaps(data, name="renewables_generated_all")

    # data= data.fillna(0)

    _log_helper_df_stats(data, name="renewables_generated_all")

    os.makedirs('fetched_data/renewables_gen/', exist_ok=True)
    start_time_label = _safe_timestamp_for_filename(start_time)
    data.to_csv(f'fetched_data/renewables_gen/all_gen'
                f'{start_time_label}_for'
                f'{horizon}_days.csv')

    return data

def fetch_renewable_forecast(start_time: str,
                             horizon: int = 1,
                             country_code : str = "DE_LU",
                             client_type = EntsoePandasClient,
                             ) -> pd.DataFrame :

    tz  = "Europe/Brussels"

    start_time = pd.Timestamp(start_time, tz=tz)
    end_time = start_time + pd.Timedelta(horizon, unit='D')
    client = client_type(API_KEY)
    data = client.query_wind_and_solar_forecast(country_code=country_code, start=start_time, end=end_time)
    data.index.name = 'time'
    data = _align_to_expected_index(data,
                                    start_time=start_time,
                                    end_time=end_time,
                                    name="renewables_forecast")
    data = _interpolate_small_gaps(data, name="renewables_forecast")

    _log_helper_df_stats(data,
                         name="renewables_forecast",
                         start_time=start_time,
                         end_time=end_time,
                         logger_file="fetch_forecast"

                         )

    data.columns = ['Solar_forecast','Wind_Offshore_forecast','Wind_Onshore_forecast']

    os.makedirs('fetched_data/renewables_forecast/', exist_ok=True)
    data.to_csv(f'fetched_data/renewables_forecast/forecast_'
                f'{start_time.strftime("%Y-%m-%d")}_to_'
                f'{end_time.strftime("%Y-%m-%d")}.csv')

    return data


def fetch_loads(start_time: str,
                horizon: int = 1,
                country_code : str = "DE_LU",
                client_type = EntsoePandasClient,
                ) -> pd.DataFrame :


    tz  = "Europe/Brussels"
    start_time = pd.Timestamp(start_time, tz=tz)
    end_time = start_time + pd.Timedelta(horizon, unit='D')

    client = client_type(API_KEY)
    data = client.query_load_and_forecast(country_code=country_code, start=start_time, end=end_time)
    data.index.name = 'time'
    data = _align_to_expected_index(data,
                                    start_time=start_time,
                                    end_time=end_time,
                                    name="loads")
    data = _interpolate_small_gaps(data, name="loads")

    _log_helper_df_stats(data,
                         name="loads",
                         start_time=start_time,
                         end_time=end_time,
                         logger_file="fetch_loads",
                         )

    os.makedirs('fetched_data/loads/', exist_ok=True)
    loads_output = (f'fetched_data/loads/load_'
    f'{start_time.strftime("%Y-%m-%d")}_to_'
    f'{end_time.strftime("%Y-%m-%d")}.csv'
                    )

    data.to_csv(loads_output)
    return data


def fetch_day_ahead(start_time: str,
                             horizon: int = 1,
                             country_code : str = "DE_LU",
                             client_type = EntsoePandasClient,
                             ) -> pd.DataFrame :


    tz  = "Europe/Brussels"

    start_time = pd.Timestamp(start_time, tz=tz)
    end_time = start_time + pd.Timedelta(horizon, unit='D')
    client = client_type(API_KEY)
    data = client.query_day_ahead_prices(country_code=country_code, start=start_time, end=end_time)


    ## remove the last hour

    data = data.to_frame('day_ahead_price')
    data.index.name = 'time'
    timegap = (int((data.index[1] - data.index[0]) / pd.Timedelta('1min')))

    if timegap != 15:
        # treat as hourly price and fill backwards:
        data = data.resample(rule = '15min').ffill()
        data = data.iloc[:-1,:]
    else:
        data = data.iloc[:-1,:]
    data = _align_to_expected_index(data,
                                    start_time=start_time,
                                    end_time=end_time,
                                    name="day_ahead_price")
    data = data.ffill().bfill()

    _log_helper_df_stats(data,
                         name="day_ahead_price",
                         start_time=start_time,
                         end_time=end_time,
                         logger_file="fetch_day_ahead",
                         timegap = timegap
                         )

    os.makedirs('fetched_data/day_ahead_price/', exist_ok=True)

    ## check for outliers, data issue
    price_mean = data['day_ahead_price'].mean()
    data['day_ahead_price'] = data['day_ahead_price'].where(
        data['day_ahead_price'] > -150,
        price_mean
    )

    data.to_csv(f'fetched_data/day_ahead_price/'
                f'{start_time.strftime("%Y-%m-%d")}_to_'
                f'{end_time.strftime("%Y-%m-%d")}.csv')
    return data



def data_merger (df_price:pd.DataFrame,
                 df_load:pd.DataFrame,
                 df_gen_actual:pd.DataFrame,
                 df_renewable_forecast:pd.DataFrame
                 )-> tuple:

    _assert_same_index(df_price, df_load, "price", "load")
    _assert_same_index(df_gen_actual, df_renewable_forecast, "generation", "renewable forecast")
    _assert_same_index(df_price, df_renewable_forecast, "price", "renewable forecast")

    merged_data = pd.concat([df_price, df_load, df_gen_actual, df_renewable_forecast], axis=1)

    _log_helper_df_stats(merged_data,
                         name="merged_data",
                         logger_file="data_merger")

    os.makedirs('fetched_data/merged/', exist_ok=True)

    output_path = (
        f'fetched_data/merged/merged_'
        f'{merged_data.index[1].strftime("%Y-%m-%d")}_to_'
        f'{merged_data.index[-1].strftime("%Y-%m-%d")}.csv'
    )

    merged_data.to_csv(output_path)

    return merged_data, output_path


def inference_data_merger(df_price: pd.DataFrame,
                          df_load: pd.DataFrame,
                          df_renewable_forecast: pd.DataFrame) -> tuple:

    _assert_same_index(df_price, df_load, "price", "load")
    _assert_same_index(df_price, df_renewable_forecast, "price", "renewable forecast")

    merged_data = pd.concat([df_price, df_load, df_renewable_forecast], axis=1)

    _log_helper_df_stats(merged_data,
                         name="inference_merged_data",
                         logger_file="inference_data_merger")

    os.makedirs('fetched_data/merged/', exist_ok=True)

    output_path = (
        f'fetched_data/merged/merged_'
        f'{merged_data.index[1].strftime("%Y-%m-%d")}_to_'
        f'{merged_data.index[-1].strftime("%Y-%m-%d")}.csv'
    )

    merged_data.to_csv(output_path)

    return merged_data, output_path



if __name__ == '__main__':

    start_time = "2024-01-01"
    horizon   = 2*365 # days
    assets = ['B16', 'B18','B19']

    df_concat_gen_renewables = concat_gen_data(start_time=start_time,
                                               horizon=horizon,
                                              )

    df_price_day_ahead = fetch_day_ahead(start_time=start_time,
                                         horizon=horizon,
                                         )
    df_forecast_renewables = fetch_renewable_forecast(start_time=start_time,
                                                      horizon=horizon)
    df_loads = fetch_loads(start_time=start_time,
                           horizon=horizon)
    merged, output_path = data_merger(df_price=df_price_day_ahead,
                         df_load= df_loads,
                         df_gen_actual=df_concat_gen_renewables,
                         df_renewable_forecast= df_forecast_renewables)
