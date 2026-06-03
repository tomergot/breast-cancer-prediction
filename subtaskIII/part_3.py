import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from scikeras.wrappers import KerasRegressor
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# --- קריאת הנתונים ---
X_df = pd.read_csv("cleaned.csv", low_memory=False)
y_df = pd.read_csv("train.labels.1.csv")
y = y_df["אבחנה-Tumor size"]

# --- הסרת הפיצ'ר side אם קיים ---
if "side" in X_df.columns:
    X_df = X_df.drop(columns=["side"])
if "form_name" in X_df.columns:
    X_df = X_df.drop(columns=["form_name"])

# --- המרת טקסט למספרים וניקוי ---
for col in X_df.columns:
    if X_df[col].dtype == "object":
        try:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        except:
            X_df[col] = X_df[col].astype("category").cat.codes
X_df = X_df.fillna(X_df.mean(numeric_only=True))

# --- אימפיוטר, סקלר ---
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# --- דגימה אם צריך ---
X_sample = X_scaled[:5000]
y_sample = y[:5000]
feature_names = X_df.columns

# --- מודל 1: XGBoost ---
xgb_model = XGBRegressor(n_estimators=100, objective='reg:squarederror')
xgb_model.fit(X_sample, y_sample)
xgb_importance = xgb_model.feature_importances_

# --- מודל 2: LassoCV ---
lasso = LassoCV(cv=5).fit(X_sample, y_sample)
lasso_coefs = np.abs(lasso.coef_)

# --- מודל 3: RidgeCV ---
ridge = RidgeCV(cv=5).fit(X_sample, y_sample)
ridge_coefs = np.abs(ridge.coef_)

# --- מודל 4: PCA ---
pca = PCA(n_components=min(20, X_sample.shape[1]))
pca.fit(X_sample)
pca_components = np.sum(np.abs(pca.components_), axis=0)

# --- מודל 5: רשת נוירונים עם Permutation Importance ---
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_sample.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

nn_model = KerasRegressor(model=build_model, epochs=30, batch_size=32, verbose=0)
nn_model.fit(X_sample, y_sample)
perm_importance = permutation_importance(nn_model, X_sample, y_sample, n_repeats=5, random_state=0)
nn_importance = perm_importance.importances_mean

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# --- מודל 6: Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_sample, y_sample)
rf_importance = rf_model.feature_importances_

# --- מודל 7: AdaBoost ---
adb_model = AdaBoostRegressor(n_estimators=100, random_state=42)
adb_model.fit(X_sample, y_sample)
adb_importance = adb_model.feature_importances_


# --- פונקציית הדפסת טופ פיצ'רים ---
def print_top_features(name, values, top_n=15):
    print(f"\n--- {name} ---")
    top_idx = np.argsort(values)[::-1][:top_n]
    for i in top_idx:
        print(f"{feature_names[i]}: {values[i]:.4f}")
    return pd.Series(values[top_idx], index=feature_names[top_idx], name=name)

# --- הדפסת תוצאות ---
df_compare = pd.concat([
    print_top_features("XGBoost", xgb_importance),
    print_top_features("RandomForest", rf_importance),
    print_top_features("AdaBoost", adb_importance),
    print_top_features("Lasso", lasso_coefs),
    print_top_features("Ridge", ridge_coefs),
    print_top_features("PCA", pca_components),
    print_top_features("NeuralNet", nn_importance)
], axis=1)

# הצגת התוצאה
df_compare.to_csv("feature_importance_comparison_part3.csv", index=False)
print("\n--- טבלת השוואת חשיבויות נשלחה לקובץ feature_importance_comparison_part3.csv ---")
