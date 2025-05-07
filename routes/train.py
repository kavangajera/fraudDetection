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
    # Apply exponential transformation to emphasize high anomaly scores
    # This will push higher scores (likely frauds) closer to 1
    transformed = np.exp(scores * 2) / np.exp(np.max(scores) * 2)
    
    # Further boost separation by applying a power transformation
    boosted = np.power(transformed, 1.5)
    
    # Rescale to [0,1]
    rescaled = (boosted - np.min(boosted)) / (np.max(boosted) - np.min(boosted))
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
            
            # Prepare feature matrix
            X = df.drop(columns=["a.accountNumber", "a.is_fraud", "a.fraud_score"], errors="ignore")
            
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
            
            # Calculate scores and transform
            raw_scores = -model.decision_function(X_scaled)
            fraud_scores = custom_fraud_transformation(raw_scores)
            
            # Add scores to dataframe
            df["fraud_score"] = fraud_scores
            
            # Save results
            scored_csv_path = os.path.join(UPLOAD_FOLDER, 'scored_fraud_data_optimized.csv')
            model_path = os.path.join(MODEL_FOLDER, 'model', 'isolation_forest_model_fraud_focused.pkl')
            scaler_path = os.path.join(MODEL_FOLDER, 'scaler', 'scaler_fraud_focused.pkl')
            
            df.to_csv(scored_csv_path, index=False)
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Create response array
            response_data = [
                {"accountNo": acc, "fraudScore": score} 
                for acc, score in zip(df["a.accountNumber"], df["fraud_score"])
            ]
            
            return jsonify(response_data), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500