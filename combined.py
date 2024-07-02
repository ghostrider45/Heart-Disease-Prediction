from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
app = Flask(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
# Load your dataset and perform data preprocessing
file_path = "./hartinfo.csv"
hart = pd.read_csv(file_path)

X = hart.drop(columns="target", axis=1)
Y = hart["target"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

@app.route('/')
def index():
    return render_template('index1.html')

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

        # Make predictions
        prediction = model.predict(input_data_reshaped)

        if prediction[0] == 0:
            result = "Person Doesn't have heart Disease"
        else:
            result = "Person Has Heart Disease"

        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", str(e))  # Print the error message to the console
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

