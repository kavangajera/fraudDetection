import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
# Load the trained model
model_path = "isolation_forest_model.pkl"
model = joblib.load(model_path)

# Assume the scaler used for training is needed as well
# If not saved, we'll retrain a new MinMaxScaler on the original training data
training_data = pd.read_csv("dummy_fraud_graph_features.csv")
X_train = training_data.drop(columns=["is_fraud", "fraud_score"], errors="ignore")

# Fit scaler on training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define 5 test transactions (same as above)
test_data = pd.DataFrame([
    {
        "pagerank": 0.32,
        "degree": 9,
        "betweenness": 0.048,
        "community_size": 55,
        "triangle_count": 16,
        "cycle_count": 2,
        "intermediate_accounts": 1.8
    },
    {
        "pagerank": 0.62,
        "degree": 26,
        "betweenness": 0.21,
        "community_size": 4,
        "triangle_count": 0,
        "cycle_count": 11,
        "intermediate_accounts": 9
    },
    {
        "pagerank": 0.29,
        "degree": 12,
        "betweenness": 0.05,
        "community_size": 48,
        "triangle_count": 18,
        "cycle_count": 3,
        "intermediate_accounts": 2.5
    },
    {
        "pagerank": 0.58,
        "degree": 22,
        "betweenness": 0.18,
        "community_size": 6,
        "triangle_count": 2,
        "cycle_count": 8,
        "intermediate_accounts": 7
    },
    {
        "pagerank": 0.35,
        "degree": 8,
        "betweenness": 0.045,
        "community_size": 52,
        "triangle_count": 15,
        "cycle_count": 2,
        "intermediate_accounts": 2
    }
])

# Scale test data
X_test_scaled = scaler.transform(test_data)

# Predict raw anomaly scores
raw_scores = -model.decision_function(X_test_scaled)

# Normalize to fraud scores in [0, 1]
fraud_scores = MinMaxScaler().fit_transform(raw_scores.reshape(-1, 1)).flatten()
test_data["fraud_score"] = fraud_scores

print(test_data)
