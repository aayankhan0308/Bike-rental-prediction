import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
print("✅ Data loaded:", df.shape)

# ── 2. PREPROCESSING ──────────────────────────────────────────
# Drop date column
df.drop(columns=['Date'], inplace=True)

# Encode categorical columns
le = LabelEncoder()
df['Seasons']        = le.fit_transform(df['Seasons'])
df['Holiday']        = le.fit_transform(df['Holiday'])
df['Functioning Day']= le.fit_transform(df['Functioning Day'])

print("✅ Preprocessing done")
print("Columns:", df.columns.tolist())

# ── 3. SPLIT FEATURES & TARGET ────────────────────────────────
X = df.drop(columns=['Rented Bike Count'])
y = df['Rented Bike Count']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# ── 4. TRAIN MODEL ────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── 5. EVALUATE ───────────────────────────────────────────────
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 R² Score : {r2:.4f}")
print(f"📊 RMSE     : {rmse:.2f}")

# ── 6. GRAPH 5 — Actual vs Predicted ─────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual', color='steelblue')
plt.plot(y_pred[:100],        label='Predicted', color='tomato')
plt.title('Actual vs Predicted Bike Count (First 100 samples)')
plt.xlabel('Sample')
plt.ylabel('Bike Count')
plt.legend()
plt.tight_layout()
plt.savefig('graph5_actual_vs_predicted.png')
plt.show()
print("✅ Graph 5 saved")

# ── 7. GRAPH 6 — Feature Importance ──────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='bar', color='steelblue')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('graph6_feature_importance.png')
plt.show()
print("✅ Graph 6 saved")

# ── 8. SAVE MODEL ─────────────────────────────────────────────
with open('bike_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved as bike_model.pkl")

print("\n🎉 ALL DONE! Your model is ready.")