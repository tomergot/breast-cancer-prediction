import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# --- שלב 1: עיבוד פיצ'רים ---
def prepare_features(X_df, y_series=None, k_features=None):
    numeric_df = X_df.select_dtypes(include=['int64', 'float64']).copy()
    numeric_df = numeric_df ** 2

    to_encode = [col for col in X_df.columns if X_df[col].dtype == 'object' and X_df[col].nunique() <= 10]
    onehot_df = pd.get_dummies(X_df[to_encode], drop_first=True)

    combined = pd.concat([numeric_df, onehot_df], axis=1)
    imputed = SimpleImputer(strategy='mean').fit_transform(combined)

    if k_features is not None and y_series is not None:
        selector = SelectKBest(score_func=f_regression, k=min(k_features, imputed.shape[1]))
        return selector.fit_transform(imputed, y_series)

    return imputed

# --- שלב 2: פיצול נתונים ---
def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- שלב 3: שילוב תחזיות XGBoost כווקטור פיצ'ר ---
def append_xgb_predictions(X_train, X_val, X_test, y_train, y_val):
    X_combined = np.concatenate([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        verbosity=0,
        random_state=42
    )

    xgb_model.fit(X_combined, y_combined)

    xgb_train_preds = xgb_model.predict(X_train).reshape(-1, 1)
    xgb_val_preds = xgb_model.predict(X_val).reshape(-1, 1)
    xgb_test_preds = xgb_model.predict(X_test).reshape(-1, 1)

    X_train_aug = np.hstack([X_train, xgb_train_preds])
    X_val_aug = np.hstack([X_val, xgb_val_preds])
    X_test_aug = np.hstack([X_test, xgb_test_preds])

    return X_train_aug, X_val_aug, X_test_aug, xgb_model

# --- שלב 4: אימון רשת נוירונים על הפיצ'רים המשולבים ---
def train_and_evaluate_nn(X_train, X_val, X_test, y_train, y_val, y_test):
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True
    )

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print("\n--- Validation Results ---")
    print("MSE:", mean_squared_error(y_val, y_val_pred))
    print("R²:", r2_score(y_val, y_val_pred))

    print("\n--- Test Results ---")
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("R²:", r2_score(y_test, y_test_pred))

    # שמירת תחזיות ונתוני אימון
    pd.DataFrame({"Predicted Tumor Size": y_test}).to_csv("val_predictions_part2.csv", index=False)
    print("\n✓ Saved predictions to val_predictions_part2.csv")
    pd.DataFrame({"Predicted Tumor Size": y_test_pred}).to_csv("train_predictions_part2.csv", index=False)
    print("\n✓ Saved predictions to train_predictions_part2.csv")

    return model

# --- שלב 5: הרצה סופית כולל חיזוי על cleaned_test.csv ---
def predict_on_cleaned_test(model, xgb_model, cleaned_test_path, feature_names, imputer, scaler):
    # שלב 1: קריאה ועיבוד כמו ב-train
    X_test_df = pd.read_csv(cleaned_test_path, low_memory=False)

    # עיבוד פיצ'רים
    numeric_df = X_test_df.select_dtypes(include=['int64', 'float64']) ** 2
    cat_cols = [col for col in X_test_df.columns if X_test_df[col].dtype == 'object' and X_test_df[col].nunique() <= 10]
    onehot_df = pd.get_dummies(X_test_df[cat_cols], drop_first=True)

    combined = pd.concat([numeric_df, onehot_df], axis=1)

    # השלמת עמודות חסרות
    for col in feature_names:
        if col not in combined:
            combined[col] = 0

    # סדר נכון
    combined = combined[feature_names]

    # שימוש באותם אימפיוטר וסקלר
    X_imputed = imputer.transform(combined)
    X_scaled = scaler.transform(X_imputed)

    # שלב 2: תחזיות XGBoost כהוספה
    xgb_preds = xgb_model.predict(X_scaled).reshape(-1, 1)
    X_augmented = np.hstack([X_scaled, xgb_preds])

    # שלב 3: תחזית סופית
    y_final_preds = model.predict(X_augmented)

    # שלב 4: שמירה
    pd.DataFrame({"Predicted Tumor Size": y_final_preds}).to_csv("predictions_part2.csv", index=False)
    print("\n✓ Saved predictions to predictions_part2.csv")

# --- שלב 6: main ---
def main(train_path, labels_path, test_path):
    X_df = pd.read_csv(train_path, low_memory=False)
    y_df = pd.read_csv(labels_path, low_memory=False)
    y_series = y_df["אבחנה-Tumor size"].copy()

    # שלב 1: עיבוד פיצ'רים ידני כדי לשמור את imputer, scaler, feature_names
    numeric_df = X_df.select_dtypes(include=['int64', 'float64']) ** 2
    cat_cols = [col for col in X_df.columns if X_df[col].dtype == 'object' and X_df[col].nunique() <= 10]
    onehot_df = pd.get_dummies(X_df[cat_cols], drop_first=True)
    combined = pd.concat([numeric_df, onehot_df], axis=1)

    feature_names = combined.columns.tolist()

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(combined)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # שלב 2: פיצול
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y_series)

    # שלב 3: תחזיות XGBoost כהוספה
    X_train_aug, X_val_aug, X_test_aug, xgb_model = append_xgb_predictions(X_train, X_val, X_test, y_train, y_val)

    # שלב 4: רשת נוירונים
    nn_model = train_and_evaluate_nn(X_train_aug, X_val_aug, X_test_aug, y_train, y_val, y_test)

    # שלב 5: חיזוי על cleaned_test.csv עם אותם אובייקטים
    predict_on_cleaned_test(
        model=nn_model,
        xgb_model=xgb_model,
        cleaned_test_path=test_path,
        feature_names=feature_names,
        imputer=imputer,
        scaler=scaler
    )

# --- הרצה ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and predict tumor size using neural network and XGBoost stacking.")
    parser.add_argument('--train', type=str, default="cleaned.csv", help="Path to the training features CSV file.")
    parser.add_argument('--labels', type=str, default="train.labels.1.csv", help="Path to the training labels CSV file.")
    parser.add_argument('--test', type=str, default="cleaned_test.csv", help="Path to the test features CSV file.")
    args = parser.parse_args()
    main(args.train, args.labels, args.test)
