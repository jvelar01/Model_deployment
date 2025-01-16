import pandas as pd
import mlflow
import joblib
import numpy as np


#l=[[np.int64(7), np.int64(4), 'oct', 'tue', np.float64(90.6), np.float64(35.4), np.float64(669.1), np.float64(6.7), np.float64(18.0), np.int64(33), np.float64(0.9), np.float64(0.0)]]


#BUENO A VER ESTAS COSAS HAY QUE HACERAS BASTANTE MEJOR ORDENADAS, AHORA VEREMOS
#nos llega una lista de listas con las cosas en plan friday y tal

mlflow.set_tracking_uri("http://127.0.0.1:8080")

run_id = "4cce2c7a3cf541d692477068f587092b"  
artifact_path = "scalers/scaler.pkl"

#load scaler
scaler_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
loaded_scaler = joblib.load(scaler_local_path)

def encodeData(l):
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    df = pd.DataFrame(l, columns=[f'X_{i+1}' for i in range(12)])


    for day in days:
        df[day] = (df['X_4'] == day).astype(int)

    for month in months:
        df[month] = (df['X_3'] == month).astype(int)

    df = df.drop(columns=['X_3', 'X_4'])

    rename_map = {
    'X_1': 'X',
    'X_2': 'Y',
    'X_5': 'FFMC',
    'X_6': 'DMC',
    'X_7': 'DC',
    'X_8': 'ISI',
    'X_9': 'temp',
    'X_10': 'RH',
    'X_11': 'wind',
    'X_12': 'rain',
    'mon': 'day_mon',
    'tue': 'day_tue',
    'wed': 'day_wed',
    'thu': 'day_thu',
    'fri': 'day_fri',
    'sat': 'day_sat',
    'sun': 'day_sun',
    'jan': 'month_jan',
    'feb': 'month_feb',
    'mar': 'month_mar',
    'apr': 'month_apr',
    'may': 'month_may',
    'jun': 'month_jun',
    'jul': 'month_jul',
    'aug': 'month_aug',
    'sep': 'month_sep',
    'oct': 'month_oct',
    'nov': 'month_nov',
    'dec': 'month_dec',
    }

    df_renamed = df.rename(columns=rename_map)

    desired_order = [
    'X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
    'month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul',
    'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
    'day_fri', 'day_mon', 'day_sat', 'day_sun', 'day_thu', 'day_tue', 'day_wed'
    ]

    df_reordered = df_renamed[desired_order]

    return df_reordered


    


def preprocess_data(df):
    columns = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
    df_log=df.copy()
    for column in columns:
        df_log[column] = np.log(df_log[column] + 1)
    df_scaled= loaded_scaler.transform(df_log)
    return df_scaled