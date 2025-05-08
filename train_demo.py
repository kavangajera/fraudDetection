import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

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

def train_watchdog_from_file(filepath):
    try:
        ensure_directories()

        if not os.path.isfile(filepath) or not allowed_file(filepath):
            raise ValueError("Invalid file path or unsupported file type.")

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
            df['is_suspicious'] = df['is_suspicious'].astype(bool)
            df.loc[df['is_suspicious'], 'fraud_score'] = np.clip(df.loc[df['is_suspicious'], 'fraud_score'] + 0.2, 0, 1)

        # Save results and models
        scored_csv_path = os.path.join(UPLOAD_FOLDER, 'scored_fraud_data_optimized_new.csv')
        model_path = os.path.join(MODEL_FOLDER, 'model', 'isolation_forest_model_fraud_focused_new.pkl')
        scaler_path = os.path.join(MODEL_FOLDER, 'scaler', 'scaler_fraud_focused_new.pkl')

        df.to_csv(scored_csv_path, index=False)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Return results
        return [
            {"accountNo": acc, "fraudScore": score}
            for acc, score in zip(df["accountNumber"], fraud_scores)
        ]

    except Exception as e:
        print(f"Error during training: {e}")
        return None
    
results = train_watchdog_from_file("data/transactions.csv")
if results:
    for r in results[:5]:  # Show first 5 results
        print(r)

