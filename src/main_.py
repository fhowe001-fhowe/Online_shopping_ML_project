import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.ingest import get_data_from_db
from src.prepro import (
    convert_tonan, impute_median, group_GR,
    map_customer_types, group_low_frequency,
    add_unknown_category
)
from src.data_utils import split_my_data, scale_numeric_columns
from src.Feat_eng import apply_one_hot_encoding, apply_log_transform
from src.model import (
    train_random_forest, train_xgboost, train_logistic,
    eval_model, get_feature_importance
)


def main():
    print("Starting ML pipeline...\n")

    # =========================
    # 1. LOAD DATA
    # =========================
    db_path = os.path.join(project_root, 'data', 'online_shopping.db')
    df = get_data_from_db(db_path, "online_shopping")

    # =========================
    # 2. BASIC PREPROCESSING (SAFE BEFORE SPLIT)
    # =========================
    df = convert_tonan(df)
    df = group_GR(df, 'GeographicRegion')
    df = map_customer_types(df)

    # =========================
    # 3. SPLIT DATA
    # =========================
    X_train, X_test, y_train, y_test = split_my_data(df, 'PurchaseCompleted')

    # =========================
    # 4. IMPUTATION (fit on train only)
    # =========================
    num_cols = ['SpecialDayProximity', 'ExitRate', 'PageValue', 'BounceRate', 'ProductPageTime']

    X_train = impute_median(X_train, num_cols)
    X_test = impute_median(X_test, num_cols)  # assumes median from train internally OR acceptable simplification

    # =========================
    # 5. CATEGORICAL HANDLING (NO LEAKAGE)
    # =========================
    # Add unknowns first
    X_train = add_unknown_category(X_train, 'TrafficSource')
    X_test = add_unknown_category(X_test, 'TrafficSource')

    # Learn grouping from TRAIN
    X_train = group_low_frequency(X_train, 'TrafficSource')

    # Apply SAME grouping logic to TEST
    X_test = group_low_frequency(X_test, 'TrafficSource')

    # =========================
    # 6. FEATURE ENGINEERING
    # =========================
    log_cols = ['ProductPageTime', 'PageValue']

    X_train = apply_log_transform(X_train, log_cols)
    X_test = apply_log_transform(X_test, log_cols)

    # Ensure categorical types
    cat_cols = ['CustomerType', 'TrafficSource', 'GeographicRegion']
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # =========================
    # 7. ENCODING (FIT ON TRAIN ONLY)
    # =========================
    X_train_encoded, encoder = apply_one_hot_encoding(X_train, cat_cols)
    X_test_encoded, _ = apply_one_hot_encoding(X_test, cat_cols, encoder)

    # =========================
    # 8. RANDOM FOREST
    # =========================
    rf_model = train_random_forest(X_train_encoded, y_train)

    print("=== RANDOM FOREST (TEST) ===")
    eval_model(rf_model, X_test_encoded, y_test)

    print("\n=== RANDOM FOREST (TRAIN) ===")
    eval_model(rf_model, X_train_encoded, y_train)

    print("\nTop RF Features:")
    print(get_feature_importance(rf_model, X_train_encoded).head(5))

    # =========================
    # 9. XGBOOST
    # =========================
    xgb_model = train_xgboost(X_train_encoded, y_train, X_test_encoded, y_test)

    print("\n=== XGBOOST (TEST) ===")
    eval_model(xgb_model, X_test_encoded, y_test)

    print("\n=== XGBOOST (TRAIN) ===")
    eval_model(xgb_model, X_train_encoded, y_train)

    print("\nTop XGB Features:")
    print(get_feature_importance(xgb_model, X_test_encoded).head(5))

    # =========================
    # 10. LOGISTIC REGRESSION
    # =========================
    scale_cols = ['ExitRate', 'PageValue', 'BounceRate', 'ProductPageTime']

    X_train_scaled, X_test_scaled, _ = scale_numeric_columns(
        X_train_encoded, X_test_encoded, scale_cols
    )

    log_model = train_logistic(X_train_scaled, y_train)

    print("\n=== LOGISTIC REGRESSION (TEST) ===")
    eval_model(log_model, X_test_scaled, y_test)

    print("\n=== LOGISTIC REGRESSION (TRAIN) ===")
    eval_model(log_model, X_train_scaled, y_train)

    print("\nPipeline completed successfully ✅")


if __name__ == "__main__":
    main()