import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load data
df = pd.read_csv("neo4j_network.csv")

# Step 2: Prepare feature matrix
X = df.drop(columns=["a.accountNumber", "a.is_fraud", "a.fraud_score"], errors="ignore")

# Step 3: Apply StandardScaler - better for detecting outliers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train Isolation Forest with optimized parameters for fraud detection
# Lower contamination to make the model more sensitive to anomalies
model = IsolationForest(
    n_estimators=300,          # Increased number of trees for better stability
    max_samples='auto',        # Use 'auto' for adaptive sampling
    contamination=0.05,        # Reduced contamination to increase sensitivity
    max_features=1.0,          # Use all features
    bootstrap=True,            # Use bootstrap sampling for robustness
    n_jobs=-1,                 # Use all available cores
    random_state=42,
    verbose=0
)

model.fit(X_scaled)

# Step 5: Calculate raw anomaly scores
raw_scores = -model.decision_function(X_scaled)  # Higher = more anomalous

# Step 6: Apply custom transformation to better separate fraud from non-fraud
# This function will push more confident fraud predictions toward 1.0
# and compress the lower end of the scale
def custom_fraud_transformation(scores):
    # Apply exponential transformation to emphasize high anomaly scores
    # This will push higher scores (likely frauds) closer to 1
    transformed = np.exp(scores * 2) / np.exp(np.max(scores) * 2)
    
    # Further boost separation by applying a power transformation
    boosted = np.power(transformed, 1.5)
    
    # Rescale to [0,1]
    rescaled = (boosted - np.min(boosted)) / (np.max(boosted) - np.min(boosted))
    return rescaled

# Apply custom transformation
fraud_scores = custom_fraud_transformation(raw_scores)

# Step 7: Append scores to dataframe
df["fraud_score"] = fraud_scores

# Step 8: View distribution of scores
print("Score distribution statistics:")
print(df["fraud_score"].describe())

# Step 9: View score distribution for fraud vs non-fraud if labels available
if "a.is_fraud" in df.columns:
    fraud_df = df[df["a.is_fraud"] == 1]
    non_fraud_df = df[df["a.is_fraud"] == 0]
    
    print("\nFraud score distribution:")
    print(f"Fraud accounts (min, median, max): {fraud_df['fraud_score'].min():.4f}, "
          f"{fraud_df['fraud_score'].median():.4f}, {fraud_df['fraud_score'].max():.4f}")
    print(f"Non-fraud accounts (min, median, max): {non_fraud_df['fraud_score'].min():.4f}, "
          f"{non_fraud_df['fraud_score'].median():.4f}, {non_fraud_df['fraud_score'].max():.4f}")
    
    # Calculate detection rates at different thresholds
    print("\nFraud detection performance:")
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        detected_frauds = sum(fraud_df['fraud_score'] >= threshold)
        false_positives = sum(non_fraud_df['fraud_score'] >= threshold)
        
        detection_rate = detected_frauds / len(fraud_df) * 100
        false_positive_rate = false_positives / len(non_fraud_df) * 100
        
        print(f"Threshold {threshold:.1f}: Detected {detected_frauds}/{len(fraud_df)} frauds "
              f"({detection_rate:.1f}%), False positives: {false_positive_rate:.1f}%")

# Step 10: View top suspicious records
print("\nTop suspicious accounts:")
print(df.sort_values(by="fraud_score", ascending=False).head(10))

# Save results
df.to_csv("scored_fraud_data_optimized.csv", index=False)
joblib.dump(model, "isolation_forest_model_fraud_focused.pkl")
joblib.dump(scaler, "scaler_fraud_focused.pkl")

# Optional: Add a recommended threshold based on the analysis
if "a.is_fraud" in df.columns:
    # Find threshold that gives at least 95% fraud detection
    for threshold in np.arange(0.1, 0.9, 0.05):
        detection_rate = sum(fraud_df['fraud_score'] >= threshold) / len(fraud_df)
        if detection_rate >= 0.95:
            recommended_threshold = threshold
            break
    else:
        recommended_threshold = 0.3  # Fallback if 95% can't be achieved
    
    print(f"\nRecommended threshold for 95%+ fraud detection: {recommended_threshold:.2f}")