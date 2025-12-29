import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow
from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def train_model():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Cancer_Severity_Score_Prediction")
    mlflow.autolog()
    
    print("Starting Model Training Pipeline")
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "MLProject/global_cancer_patients_2015_2024_preprocessing.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    X = df.drop(columns=['Target_Severity_Score'])
    y = df['Target_Severity_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name='Linear_Regression_Model'):
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Linear_Regression_Model",
            input_example=X_train.head()
        )

        print(f"Model trained with R2 Score: {r2} and MSE: {mse}")

if __name__=="__main__":
    train_model()
