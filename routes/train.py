import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


from flask import Blueprint, jsonify, request
train_bp = Blueprint('train', __name__)


import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'data'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_directories():
    os.makedirs(os.path.join(MODEL_FOLDER, 'model'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_FOLDER, 'scaler'), exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@train_bp.route('/', methods=['POST'])
def train_watchdog():
    try:
        ensure_directories()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Load data
            df = pd.read_csv(filepath)

            # Drop non-feature columns
            drop_cols = [
                'accountNumber', 'type', 'user',
                'is_suspicious', 'suspicious', 'fraudScore'
            ]
            X = df.drop(columns=drop_cols, errors='ignore')

            # Apply StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train Isolation Forest
            model = IsolationForest(
                n_estimators=300,
                max_samples='auto',
                contamination=0.05,
                max_features=1.0,
                bootstrap=True,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )

            model.fit(X_scaled)

            # Score and transform
            raw_scores = -model.decision_function(X_scaled)
            fraud_scores = custom_fraud_transformation(raw_scores)

            # Add fraud scores
            df['fraud_score'] = fraud_scores

            # Boost score if is_suspicious is True
            if 'is_suspicious' in df.columns:
                # Ensure it's boolean
                df['is_suspicious'] = df['is_suspicious'].astype(bool)
                
                # Define how much to boost (e.g., add 0.2, then clip to max 1.0)
                df.loc[df['is_suspicious'], 'fraud_score'] = np.clip(df.loc[df['is_suspicious'], 'fraud_score'] + 0.2, 0, 1)


            # Save results and models
            scored_csv_path = os.path.join(UPLOAD_FOLDER, 'scored_fraud_data_optimized_new.csv')
            model_path = os.path.join(MODEL_FOLDER, 'model', 'isolation_forest_model_fraud_focused_new.pkl')
            scaler_path = os.path.join(MODEL_FOLDER, 'scaler', 'scaler_fraud_focused_new.pkl')

            df.to_csv(scored_csv_path, index=False)
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)

            # Prepare response
            response_data = [
                {"accountNo": acc, "fraudScore": score}
                for acc, score in zip(df["accountNumber"], fraud_scores)
            ]

            return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
