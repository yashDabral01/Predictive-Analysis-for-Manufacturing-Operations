from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Initialize the Flask app
app = Flask(__name__)

# Error handling for file loading
try:
    preprocessing_pipeline = pickle.load(open("C:/Users/Yash Dabral/OneDrive/Desktop/Internship task/Processing/preprocessing_pipeline.pkl", 'rb'))
    model = pickle.load(open("C:/Users/Yash Dabral/OneDrive/Desktop/Internship task/Processing/voting_classifier_model.pkl", 'rb'))
except Exception as e:
    print(f"Error loading model or pipeline: {e}")

# Storage for uploaded data
uploaded_data = None

@app.route('/')
def home():
    return "Voting Classifier API is running!"

@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_data
    try:
        # Retrieve the uploaded file
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Load the CSV file into a DataFrame
        uploaded_data = pd.read_csv(file)
        return jsonify({'message': 'File uploaded successfully', 'columns': uploaded_data.columns.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    global uploaded_data, model
    try:
        if uploaded_data is None:
            return jsonify({'error': 'No data uploaded. Use the /upload endpoint first.'}), 400
       
        features=["Torque(Nm)","Hydraulic_Pressure(bar)","Cutting(kN)",
          "Coolant_Pressure(bar)","Spindle_Speed(RPM)","Coolant_Temperature"]
        # Split the data into features and target
        X = uploaded_data[features] 
        y = uploaded_data['Downtime']

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply preprocessing
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)

        # Train the model
        model.fit(X_train_processed, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        return jsonify({'message': 'Model trained successfully', 'accuracy': accuracy, 'f1_score': f1})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request
        data = request.get_json()

        # Extract the features from the input data
        features = data['features']

        # Check if the number of features is correct
        if len(features) != 6:  # Assuming only two features: Temperature and Run_Time
            return jsonify({'error': 'Incorrect number of features provided'}), 400

        # Convert input data into a numpy array (ensure it's in 2D)
        input_data = np.array(features).reshape(1, -1)

        # Apply preprocessing
        processed_data = preprocessing_pipeline.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(processed_data)

        # Convert prediction to a Yes/No response
        result = 'Yes' if prediction[0] == 1 else 'No'

        return jsonify({'Downtime': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Run the app
    app.run(host='127.0.0.1', port=5000, debug=True)
