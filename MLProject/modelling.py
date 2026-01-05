import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import mlflow
import mlflow.sklearn
import os

# Menghubungkan ke DagsHub secara otomatis di GitHub Actions
mlflow.set_tracking_uri("https://dagshub.com/putri_ackerman/my-first-repo.mlflow")

# load data
df = pd.read_csv('house_price_preprocessing/house_price_clean.csv')
X = df.drop('House_Price', axis=1) 
y = df['House_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# activate autolog
mlflow.sklearn.autolog()

# training model
rf_model = RandomForestRegressor(random_state=42)

with mlflow.start_run(run_name="Random Forest Basic"):
    rf_model.fit(X_train, y_train)
    
    # evaluasi sederhana
    y_pred = rf_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
