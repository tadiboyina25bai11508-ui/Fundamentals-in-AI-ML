import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
import joblib
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Crop Recommendation for Indian Farmers (Bhopal/MP)")
print("=" * 60)

url = "https://raw.githubusercontent.com/ATHARVAINGLE/crop-recommendation-dataset/master/Crop_recommendation.csv"
df = pd.read_csv(url)

print(f"Dataset: {df.shape}")
print(f"Crops: {df['label'].nunique()}")

fig, axes = plt.subplots(2, 3, figsize=(20, 15))

sns.countplot(y='label', data=df, order=df['label'].value_counts().index, ax=axes[0,0])
axes[0,0].set_title('Crop Distribution')

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
axes[0,1].set_title('Correlation Heatmap')

top_crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas']
df_top = df[df['label'].isin(top_crops)]
sns.scatterplot(data=df_top, x='N', y='P', hue='label', alpha=0.6, s=50, ax=axes[0,2])
axes[0,2].set_title('N vs P (Top Crops)')

sns.scatterplot(data=df_top, x='temperature', y='humidity', hue='label', alpha=0.6, s=50, ax=axes[1,0])
axes[1,0].set_title('Temp vs Humidity')

sns.boxplot(data=df, x='label', y='ph', order=df['label'].value_counts().index[:10], ax=axes[1,1])
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].set_title('Soil pH')

sns.boxplot(data=df, x='label', y='rainfall', order=df['label'].value_counts().index[:10], ax=axes[1,2])
axes[1,2].tick_params(axis='x', rotation=45)
axes[1,2].set_title('Rainfall')

plt.tight_layout()
plt.show()

X = df.drop('label', axis=1)
y = LabelEncoder().fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\nRandom Forest Accuracy: {rf_acc:.1%}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices][:10], y=X.columns[indices][:10], ax=ax1)
ax1.set_title('Top Features')

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()

print(classification_report(y_test, rf_pred, target_names=LabelEncoder().fit(df['label']).classes_))

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred = rf_model.predict(input_data)[0]
    conf = np.max(rf_model.predict_proba(input_data)) * 100
    crop = LabelEncoder().fit(df['label']).inverse_transform([pred])[0]
    return crop, conf

print("\n" + "="*50)
print("Test your farm conditions:")

print("Monsoon: ", recommend_crop(90, 42, 43, 23.2, 80.4, 6.5, 150))
print("Rabi:    ", recommend_crop(20, 50, 60, 22.0, 40.2, 7.2, 60))
print("Kharif:  ", recommend_crop(80, 40, 50, 28.5, 65.3, 6.8, 120))

joblib.dump(rf_model, 'crop_model.pkl')
print("\nModel saved as 'crop_model.pkl'")

print("\nFor Streamlit app, create app.py:")
print('''
import streamlit as st
import joblib
import numpy as np

model = joblib.load("crop_model.pkl")
le = joblib.load("crop_encoder.pkl") 

st.title("Crop Recommender")

col1, col2 = st.columns(2)
N = col1.slider("Nitrogen", 0, 200, 50)
P = col1.slider("Phosphorus", 0, 200, 50)
K = col1.slider("Potassium", 0, 200, 50)
temp = col2.slider("Temperature", 0.0, 50.0, 25.0)
hum = col2.slider("Humidity", 0, 100, 60)
ph = col2.slider("pH", 0.0, 14.0, 6.5)
rain = col2.slider("Rainfall", 0, 300, 100)

if st.button("Recommend"):
    pred = model.predict([[N,P,K,temp,hum,ph,rain]])[0]
    st.success(f"{le.inverse_transform([pred])[0].upper()}")
''')

print("\nRun: streamlit run app.py")
print(" Done!")