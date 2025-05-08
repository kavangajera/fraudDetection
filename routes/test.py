import os
import requests
import pandas as pd
import numpy as np
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
import joblib

test_bp = Blueprint('test', __name__)

MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def custom_fraud_transformation(scores):
    # Apply exponential transformation to emphasize high anomaly scores
    transformed = np.exp(scores * 2) / np.exp(np.max(scores) * 2)
    
    # Further boost separation by applying a power transformation
    boosted = np.power(transformed, 1.5)
    
    # Rescale to [0,1]
    rescaled = (boosted - np.min(boosted)) / (np.max(boosted) - np.min(boosted))
    return rescaled

@test_bp.route('/', methods=['GET'])
def test_watchdog():
    try:
        

            # Load model and scaler
            model_path = os.path.join(MODEL_FOLDER, 'model', 'isolation_forest_model_fraud_focused_new.pkl')
            scaler_path = os.path.join(MODEL_FOLDER, 'scaler', 'scaler_fraud_focused_new.pkl')
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Load CSV and preprocess
            df = pd.read_csv('data/acc_list_with_params.csv')

            # Drop non-feature columns
            drop_cols = [
                'accountNumber', 'type', 'user', 
                'is_suspicious', 'suspicious', 'fraudScore'
            ]
            X_test = df.drop(columns=drop_cols, errors='ignore')

            # Scale and predict
            X_scaled = scaler.transform(X_test)
            raw_scores = -model.decision_function(X_scaled)
            fraud_scores = custom_fraud_transformation(raw_scores)

            # Append prediction to original dataframe
            df['calculated_fraud_score'] = fraud_scores

            # Append prediction to original dataframe
            df['calculated_fraud_score'] = fraud_scores

            # Boost score if is_suspicious is True
            if 'is_suspicious' in df.columns:
                df['is_suspicious'] = df['is_suspicious'].astype(bool)
                df.loc[df['is_suspicious'], 'calculated_fraud_score'] = np.clip(
                    df.loc[df['is_suspicious'], 'calculated_fraud_score'] + 0.2, 0, 1
                )


            # Save result to CSV
            output_path = os.path.join('data', 'result.csv')
            os.makedirs('data', exist_ok=True)
            df.to_csv(output_path, index=False)

            # Prepare response
            # Prepare response with camelCase keys (Java convention)
            response = [
                {
                    "accountNumber": str(acc),  # ensure JSON-safe string if needed
                    "calculatedFraudScore": float(score)  # ensure JSON serializable
                }
                for acc, score in zip(df["accountNumber"], df["calculated_fraud_score"])
            ]

            return jsonify({"accounts": response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
