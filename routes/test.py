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

@test_bp.route('/', methods=['POST'])
def test_watchdog():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            # Save file temporarily to read it twice
            temp_file_path = os.path.join('temp', secure_filename(file.filename))
            os.makedirs('temp', exist_ok=True)
            file.save(temp_file_path)
            
            # Load model and scaler
            model_path = os.path.join(MODEL_FOLDER, 'model', 'isolation_forest_model_fraud_focused.pkl')
            scaler_path = os.path.join(MODEL_FOLDER, 'scaler', 'scaler_fraud_focused.pkl')
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Read and process test data
            df = pd.read_csv(temp_file_path)
            X_test = df.drop(columns=["a.accountNumber", "a.is_fraud", "a.fraud_score"], errors="ignore")
            
            # Scale test data
            X_test_scaled = scaler.transform(X_test)
            
            # Get predictions
            raw_scores = -model.decision_function(X_test_scaled)
            fraud_scores = custom_fraud_transformation(raw_scores)
            
            # Add scores to dataframe
            df["fraud_score"] = fraud_scores
            
            # Add predicted fraud score to dataframe
            df["predicted_fraud_score"] = fraud_scores
            
            # Save to data folder
            output_path = os.path.join('data', 'modified_accounts.csv')
            df.to_csv(output_path, index=False)

            # Create response array
            response_data = [
                {"accountNo": acc, "fraudScore": score} 
                for acc, score in zip(df["a.accountNumber"], df["fraud_score"])
            ]
            
            # Define the training URL
            # train_url = 'http://localhost:5000/train'  # Replace with the actual training endpoint
            
            # # Send the file to train route for retraining in background
            # with open(temp_file_path, 'rb') as f:
            #     files = {'file': (file.filename, f)}
            #     # Non-blocking request for training
            #     try:
            #         requests.post(train_url, files=files, timeout=0.1)
            #     except requests.exceptions.ReadTimeout:
            #         pass  # This is expected, we don't wait for training response
            
            # # Clean up temp file
            # os.remove(temp_file_path)
            
            return jsonify(response_data), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500