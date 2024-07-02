import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data and preprocessing (unchanged from original code)
file_path = "combined.xlsx"
df = pd.read_excel(file_path)
df = df[df["restecg"] != "LVH"]

X = df.drop(columns=["output"])
y = df["output"]
df = df.drop_duplicates()

print("Number of duplicate rows:", df.duplicated().sum())
print("Number of missing values in each column:", df.isnull().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model performance
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}")

# Make a prediction on a sample input
sample_input = [44,1,4,120,169,0 ,0,144, 1 ,2.8,3]
sample_scaled = scaler.transform([sample_input])
sample_pred = rf_model.predict(sample_scaled)[0]

heart_conditions = { 0: "No heart disease",
1: "Cardiovascular heart disease",
2: "Heart attack", 
3: "Heart failure"
}

predicted_condition = heart_conditions.get(sample_pred, "Unknown")
print(f"Predicted heart condition: {predicted_condition}")
