import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')

# ── 2. BASIC INFO ─────────────────────────────────────────────
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nBasic stats:\n", df.describe())

# ── 3. GRAPH 1 — Hourly Bike Demand ───────────────────────────
plt.figure(figsize=(10, 5))
sns.lineplot(x='Hour', y='Rented Bike Count', data=df, errorbar=None, color='steelblue')
plt.xlabel('Hour')
plt.ylabel('Avg Bike Count')
plt.tight_layout()
plt.savefig('graph1_hourly_demand.png')
plt.show()
print("✅ Graph 1 saved")

# ── 4. GRAPH 2 — Bikes by Season ──────────────────────────────
season_map = {'Spring': 1, 'Summer': 2, 'Autumn': 3, 'Winter': 4}
df['Season Name'] = df['Seasons']
season_avg = df.groupby('Season Name')['Rented Bike Count'].mean().reset_index()
plt.figure(figsize=(7, 5))
plt.bar(season_avg['Season Name'], season_avg['Rented Bike Count'], color=['#4CAF50','#FF9800','#F44336','#2196F3'])
plt.title('Average Bike Rentals by Season')
plt.xlabel('Season')
plt.ylabel('Avg Bike Count')
plt.tight_layout()
plt.savefig('graph2_season.png')
plt.show()
print("✅ Graph 2 saved")

# ── 5. GRAPH 3 — Temperature vs Bike Count ────────────────────
plt.figure(figsize=(8, 5))
sns.boxplot(x='Season Name', y='Rented Bike Count', data=df, hue='Season Name', legend=False)
plt.title('Temperature vs Bike Rentals')
plt.xlabel('Temperature (°C)')
plt.ylabel('Bike Count')
plt.tight_layout()
plt.savefig('graph3_temperature.png')
plt.show()
print("✅ Graph 3 saved")

# ── 6. GRAPH 4 — Correlation Heatmap ─────────────────────────
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('graph4_heatmap.png')
plt.show()
print("✅ Graph 4 saved")

print("\n✅ EDA COMPLETE — 4 graphs saved in your folder!")