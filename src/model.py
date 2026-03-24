import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

def train_random_forest(X_train, y_train):
    """
    Train a well-balanced Random Forest model to avoid overfitting
    while handling class imbalance.
    """
    
    model = RandomForestClassifier(
        n_estimators=100,          # More trees → more stable
        max_depth=10,              # Prevent overfitting
        min_samples_split=5,       # Avoid splitting on tiny noise
        min_samples_leaf=2,        # Smooth predictions
        max_features='sqrt',       # Default, good for generalization
        class_weight='balanced',   # Handle imbalanced target
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train.values.ravel())
    
    return model

def eval_model(model, X_test, y_test, verbose=True):
    """
    Evaluates a classification model on test data.

    Returns:
        metrics (dict): key metrics
        y_pred (np.array): predicted labels
    """
    # Predict labels
    y_pred = model.predict(X_test)
    
    # Predict probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = None

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_probs) if y_probs is not None else None
    }

    if verbose:
        print("\n--- Model Evaluation ---")
        for k, v in metrics.items():
            if v is not None:
                print(f"{k}: {v:.4f}")
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        #print("--- Confusion Matrix ---")
        #print(confusion_matrix(y_test, y_pred))
    
    return metrics, y_pred


def get_feature_importance(model, X_train):
    importance = model.feature_importances_
    
    feat_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importance
    }).sort_values(by="importance", ascending=False)
    
    return feat_imp

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost classifier and evaluate performance.
    
    Returns:
        model: trained XGBClassifier
        metrics: dict with Accuracy, F1, AUC-ROC
        feature_importance: DataFrame with feature importances
    """    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1, #regularization
        reg_lambda=1, #regularization 
        scale_pos_weight=5, #neg/pos --> handle imbalance
        random_state=42
    )
    
    model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks =[EarlyStopping(rounds=50)],
    verbose=False
    )
    
    return model 


def train_logistic(X_train, y_train):

    model = LogisticRegression(
    penalty='l2',          # Regularization type ('l2' is default)
    C=1.0,                 # Inverse of regularization strength (smaller = stronger regularization)
    solver='lbfgs',        # Optimization algorithm
    max_iter=500,           # Max iterations to converge
    class_weight='balanced', # Adjust for imbalanced classes
    random_state=42
    )

    # Fit the model
    model.fit(X_train, y_train)

    return model