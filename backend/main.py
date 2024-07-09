from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from datetime import timedelta
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import joblib

import lightgbm as lgb


from preprocessing.preprocessing import preprocess_data

app = FastAPI()

# Global variable for the loaded model and explanation data
loaded_models = {}
models_path = Path("./models")
explain_data = {}

class PredictionResults(BaseModel):
    predictions: list
    data: str


def load_model(model_name: str):
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    model_path = models_path / model_name
    
    if model_name.endswith('.h5') or model_name.endswith('.keras'):
        # Load Keras models
        model = tf.keras.models.load_model(model_path)

    elif model_name.endswith('.joblib'):
        # Load Random Forest models
        model = joblib.load(model_path)

    elif model_name.endswith('.txt'):
        # Load LightGBM models
        model = lgb.Booster(model_file=str(model_path))
    else:
        # Raise exception if the model format is not recognized
        raise HTTPException(status_code=400, detail=f"Model format not supported: {model_name}")
    
    # Store the loaded model to avoid reloading it again
    loaded_models[model_name] = model
    return model

    




@app.post("/predict/", response_model=PredictionResults)
async def make_prediction(model_name: str, transaction_file: UploadFile = File(...), identity_file: UploadFile = File(...)):

    # Read the data files
    transaction_df = pd.read_csv(transaction_file.file)
    identity_df = pd.read_csv(identity_file.file)

    # Merge dataframes on 'TransactionID'
    master_df = pd.merge(transaction_df, identity_df, on='TransactionID', how='left')

    # Preprocess the data
    processed_data = preprocess_data(master_df)

    # Load model
    model = load_model(model_name)

    # Make predictions
    predictions = model.predict(processed_data)

    # Convert probabilities to binary predictions (0 or 1)
    binary_predictions = (np.array(predictions) > 0.5).astype(int).flatten().tolist()

    master_df['predictions'] = binary_predictions

    # Filter based on fraudulent transactions for demo
    fraudulent_predictions = master_df[master_df['predictions'] == 1]
    print(fraudulent_predictions)
    

    # Store processed data for explanations
    processed_data['TransactionID'] = master_df['TransactionID']
    explain_data[model_name] = processed_data

    # Prepare datetime for display
    START_DATE = pd.Timestamp('2017-11-30')
    master_df['TransactionDT'] = pd.to_numeric(master_df['TransactionDT'], errors='coerce')
    master_df['DT'] = START_DATE + pd.to_timedelta(master_df['TransactionDT'], unit='s')
    master_df['DT'] = master_df['DT'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Format TransactionAmt for display
    master_df['TransactionAmt'] = master_df['TransactionAmt'].apply(lambda x: f"{x:.2f}")

    output_columns = ['TransactionID', 'predictions', 'DT', 'TransactionAmt', 'ProductCD', 'card4', 'card6', 'P_emaildomain', 'DeviceType']
    return_data = master_df[output_columns].fillna('-').to_json(orient='records')
    
    return PredictionResults(predictions=binary_predictions, data=return_data)







@app.get("/explain/{transaction_id}")
async def explain_prediction(transaction_id: str, model_name: str):
    if model_name not in explain_data:
        raise HTTPException(status_code=404, detail="Model data not available for explanations.")
    
    model = load_model(model_name)
    processed_data = explain_data[model_name]
    transaction_id = int(transaction_id)
    instance_data = processed_data[processed_data['TransactionID'] == transaction_id]

    if instance_data.empty:
        raise HTTPException(status_code=404, detail="Transaction ID not found.")

    features_data = instance_data.drop(columns=['TransactionID'])
    explainer = LimeTabularExplainer(training_data=features_data.to_numpy(),
                                     feature_names=features_data.columns.tolist(),
                                     class_names=['Not Fraud', 'Fraud'],
                                     mode='classification')

    # Adjust predict function based on model output
    predict_fn = lambda x: np.column_stack([1-model.predict(x), model.predict(x)])
    exp = explainer.explain_instance(data_row=features_data.iloc[0].to_numpy(),
                                     predict_fn=predict_fn)
    return {"TransactionID": transaction_id, "explanation": exp.as_list()}































