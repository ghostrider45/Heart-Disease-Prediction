import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request, jsonify
import numpy as np
import warnings
app = Flask(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
file_path = "combined.xlsx"
df = pd.read_excel(file_path)
X = df.drop(columns=["output"])
y = df["output"]
df = df.drop_duplicates()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index2.html')
@app.route('/train', methods=['POST'])
def train_and_predict():
    try:
        input_data = request.get_json()

        # Ensure that the input_data contains all the required features
        required_features = X.columns.tolist()
        for feature in required_features:
            if feature not in input_data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert input data to the correct data types (all numeric)
        for feature in required_features:
            input_data[feature] = float(input_data[feature])

        # Convert input data to a numpy array
        input_data_as_list = [input_data[feature] for feature in required_features]
        input_data_as_numpy_array = np.asarray(input_data_as_list)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        # Make a prediction on a sample input
        
        sample_pred = rf_model.predict(input_data_reshaped)
        predicted_condition_index = int(sample_pred[0])  # Assuming sample_pred is a 1D array

        heart_conditions = {
        0: "No heart disease",
        1: "Cardiovascular heart disease",
        2: "Heart attack",
        3: "Heart failure"
        }

# Use the scalar value as the key to get the predicted condition
        predicted_condition = heart_conditions.get(predicted_condition_index, "Unknown")
        return jsonify({"prediction": predicted_condition})

    except Exception as e:
        print("Error:", str(e))  # Print the error message to the console
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)