from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def full_prepare_for_xgboost(X_df: pd.DataFrame, y_series: pd.Series = None, k_features: int = None):
    numeric_df = X_df.select_dtypes(include=['int64', 'float64']).copy()
    numeric_df = numeric_df ** 2  # ריבוע פיצ'רים נומריים

    to_encode = [col for col in X_df.columns
                 if X_df[col].dtype == 'object' and X_df[col].nunique() <= 10]

    onehot_df = pd.get_dummies(X_df[to_encode], drop_first=True)
    combined = pd.concat([numeric_df, onehot_df], axis=1)

    imputer = SimpleImputer(strategy='mean')
    combined_imputed = imputer.fit_transform(combined)

    if k_features is not None and y_series is not None:
        selector = SelectKBest(score_func=f_regression, k=min(k_features, combined_imputed.shape[1]))
        return selector.fit_transform(combined_imputed, y_series)
    return combined_imputed


def Base_Line(X_df: pd.DataFrame, y_df: pd.DataFrame, target_column: str,
              sample_size: int = 5000, k_features: int = 25):

    y_series = y_df[target_column].copy()
    y_series.index = X_df.index
    baseline_summary_sample = []

    for n_iter in [7, 15, 60, 100, 200, 250, 300, 400]:
        all_mse = []
        all_r2 = []

        for _ in range(n_iter):
            X_sample = X_df.sample(n=sample_size, random_state=None)
            y_sample = y_series.loc[X_sample.index]

            # עיבוד פיצ'רים
            X_processed = full_prepare_for_xgboost(X_sample, y_sample, k_features=k_features)
            X_scaled = StandardScaler().fit_transform(X_processed)

            # פיצול נתונים
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_sample, test_size=0.2, random_state=42
            )

            # הגדרת מודל
            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                verbosity=0
            )

            # אימון וחיזוי
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_mse.append(mean_squared_error(y_test, y_pred))
            all_r2.append(r2_score(y_test, y_pred))

        baseline_summary_sample.append({
            "Iterations": n_iter,
            "Average MSE": np.mean(all_mse),
            "Std MSE": np.std(all_mse),
            "Average R²": np.mean(all_r2),
            "Std R²": np.std(all_r2)
        })

    results_df = pd.DataFrame(baseline_summary_sample)
    print(results_df)
    return results_df


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# להבטיח שלא תיחתך הדפסה
pd.set_option("display.max_columns", None)


def Hyperparameter_BaseLine(X_df: pd.DataFrame, y_df: pd.DataFrame, target_column: str,
                            sample_size: int = 10000, k_features: int = 80,
                            n_iter: int = 30):

    y_series = y_df[target_column].copy()
    y_series.index = X_df.index

    param_grid = [
        {"name": "Baseline", "params": {
            "n_estimators": 200, "learning_rate": 0.05, "max_depth": 4,
            "reg_alpha": 0, "reg_lambda": 1, "gamma": 0
        }},
        {"name": "Regularized 1", "params": {
            "n_estimators": 300, "learning_rate": 0.03, "max_depth": 5,
            "reg_alpha": 0.5, "reg_lambda": 1.0, "gamma": 1
        }},
        {"name": "Regularized 2", "params": {
            "n_estimators": 400, "learning_rate": 0.02, "max_depth": 6,
            "reg_alpha": 1.0, "reg_lambda": 2.0, "gamma": 2
        }},
        {"name": "Light Regularized", "params": {
            "n_estimators": 250, "learning_rate": 0.05, "max_depth": 3,
            "reg_alpha": 0.1, "reg_lambda": 0.5, "gamma": 0.5
        }},
    ]

    all_results = []

    for config in param_grid:
        all_mse, all_r2 = [], []

        print(f"Running config: {config['name']}")

        for _ in range(n_iter):
            X_sample = X_df.sample(n=sample_size)
            y_sample = y_series.loc[X_sample.index]

            X_processed = full_prepare_for_xgboost(X_sample, y_sample, k_features=k_features)
            X_scaled = StandardScaler().fit_transform(X_processed)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_sample, test_size=0.2, random_state=42
            )

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
                **config["params"]
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_mse.append(mean_squared_error(y_test, y_pred))
            all_r2.append(r2_score(y_test, y_pred))

        all_results.append({
            "Model Name": config["name"],
            **config["params"],
            "Average MSE": np.mean(all_mse),
            "Std MSE": np.std(all_mse),
            "Average R²": np.mean(all_r2),
            "Std R²": np.std(all_r2)
        })

    results_df = pd.DataFrame(all_results)

    print("\n--- תוצאות מדויקות ---")
    print(results_df[["Model Name", "Average MSE", "Std MSE", "Average R²", "Std R²"]])

    best_model = results_df.loc[results_df["Average R²"].idxmax()]
    print("\n--- המודל הטוב ביותר (לפי R² ממוצע) ---")
    print(best_model[["Model Name", "Average R²", "Average MSE", "n_estimators", "learning_rate", "max_depth", "reg_alpha", "reg_lambda", "gamma"]])

    return results_df

# הפעלה לדוגמה
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run XGBoost baseline with hyperparameter search.")
    parser.add_argument('--features', type=str, default="cleaned.csv", help="Path to features CSV file")
    parser.add_argument('--labels', type=str, default="train.labels.1.csv", help="Path to labels CSV file")
    args = parser.parse_args()

    X_df = pd.read_csv(args.features, low_memory=False, na_values=["", "NA", "NaN"])
    y_df = pd.read_csv(args.labels, low_memory=False)
    Hyperparameter_BaseLine(X_df, y_df, target_column="אבחנה-Tumor size")
