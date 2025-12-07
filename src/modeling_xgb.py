import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import time

def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight=None):
    """
    Train an XGBoost classifier using SMOTENC processed dataset.

    Parameters:
        scale_pos_weight (dict or None)

    Returns:
        model: trained XGBoost model
    """

    print("\nTraining XGBoost Model...")

    params = {
        "objective": "multi:softmax",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_estimators": 350,
        "random_state": 207,
    }

    if scale_pos_weight:
        params["scale_pos_weight"] = scale_pos_weight

    start = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    duration = time.time() - start

    print(f"⏱ Training completed in {duration:.2f} seconds.\n")

    # Validation
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred))

    return model
