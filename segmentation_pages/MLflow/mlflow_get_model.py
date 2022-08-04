import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(csv_url, sep=";")
train, test = train_test_split(data)
test_x = test.drop(["quality"], axis=1)
test_y = test[["quality"]]

base_url = 'http://localhost:5000/api/2.0'
endpoint = base_url + '/mlflow/artifacts/list'

response = requests.get(endpoint, { 'run_id': '037d56ab851c4ef985b3b6a9790b0203' })

model_path = response.json()['root_uri'] + '/' + response.json()['files'][0]['path']

print(response.json())
print(model_path)

model = pd.read_pickle(model_path + '/' + 'model.pkl')

print(model)

predicted_qualities = model.predict(test_x)
(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

print(rmse, mae, r2)