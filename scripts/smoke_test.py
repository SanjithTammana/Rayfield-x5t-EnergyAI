import pandas as pd, pathlib
from energy_app.utils.predictors import forecast, classify, detect_anomaly, reduce_dim

data = pd.read_csv(pathlib.Path(__file__).parents[1] / "training_data.csv")
print("Forecast sample:", forecast("solar", data, 3))
print("Classify sample:", classify(data.head())[0][:5])
print("Anomaly flags :", detect_anomaly(data.head()))
print("Dim reduct    :", reduce_dim(data.head())[:2])