import os
import pandas as pd
import numpy as np
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

test_bp = Blueprint('test', __name__)

MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv'}

# MongoDB configuration
MONGO_URI_ACCOUNT = os.getenv('MONGO_URI_ACCOUNT')
client = MongoClient(MONGO_URI_ACCOUNT)
db = client.get_default_database()  # or client["your_db_name"]
accounts_collection = db["accounts"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def custom_fraud_transformation(scores):
    # Check if all scores are identical
    if np.max(scores) == np.min(scores):
        # If all scores are the same, return an array of 0.5
        return np.full_like(scores, 0.5)
    
    # Apply exponential transformation to emphasize high anomaly scores
    # Use a safer approach to avoid overflow
    scores_normalized = scores - np.min(scores)  # Shift to start at 0
    transformed = np.exp(scores_normalized * 2) / np.exp(np.max(scores_normalized) * 2)
    
    # Further boost separation by applying a power transformation
    boosted = np.power(transformed, 1.5)
    
    # Rescale to [0,1] with safeguard against division by zero
    range_value = np.max(boosted) - np.min(boosted)
    if range_value > 0:
        rescaled = (boosted - np.min(boosted)) / range_value
    else:
        # If all values are the same after transformation, return uniform values
        rescaled = np.full_like(boosted, 0.5)
    
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

        # Update suspiciousScore for each account
        updated_count = 0
        for _, row in df.iterrows():
            account_number = row.get("accountNumber")
            score = float(row.get("calculated_fraud_score", 0))

            if account_number:
                result = accounts_collection.update_one(
                    {"accountNumber": account_number},
                    {"$set": {"suspiciousScore": score}}
                )
                if result.modified_count > 0:
                    updated_count += 1

        # Prepare JSON response
        response = [
            {
                "accountNumber": str(acc),
                "calculatedFraudScore": float(score)
            }
            for acc, score in zip(df["accountNumber"], df["calculated_fraud_score"])
        ]

        return jsonify({
            "accounts": response,
            "message": f"{updated_count} accounts updated in MongoDB."
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
