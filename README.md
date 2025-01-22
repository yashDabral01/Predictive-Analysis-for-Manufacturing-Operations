# Predictive-Analysis-for-Manufacturing-Operations
## Voting Classifier API

This API is designed to handle file uploads, train a voting classifier model, and make predictions based on input features. It includes endpoints for uploading data, training the model, and making predictions about machine downtime.

## Setup Instructions

### Prerequisites
1. Python 3.8 or higher installed on your system.
2. Required Python libraries:
   - Flask
   - pandas
   - numpy
   - scikit-learn
   - pickle
3. Pre-trained `preprocessing_pipeline.pkl` and `voting_classifier_model.pkl` files.

### Installation Steps

1. Clone or download the project to your local machine.
2. Install the required Python packages:
   ```bash
   pip install flask pandas numpy scikit-learn
   ```
3. Place the `preprocessing_pipeline.pkl` and `voting_classifier_model.pkl` files in the appropriate :
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. By default, the application will run on `http://127.0.0.1:5000/`.

## API Endpoints

### 1. `/`
- **Method**: GET
- **Description**: Checks if the API is running.
- **Response**:
  ```json
  "Voting Classifier API is running!"
  ```

### 2. `/upload`
- **Method**: POST
- **Description**: Uploads a CSV file containing the dataset for training.
- **Request**:
  - `form-data` key: `file` (upload your CSV file)
- **Response**:
  ```json
  {
      "message": "File uploaded successfully",
      "columns": ["Torque(Nm)", "Hydraulic_Pressure(bar)", "Cutting(kN)", "Coolant_Pressure(bar)", "Spindle_Speed(RPM)", "Coolant_Temperature", "Downtime"]
  }
  ```

### 3. `/train`
- **Method**: POST
- **Description**: Trains the model using the uploaded dataset.
- **Response**:
  ```json
  {
      "message": "Model trained successfully",
      "accuracy": 0.85,
      "f1_score": 0.87
  }
  ```

### 4. `/predict`
- **Method**: POST
- **Description**: Makes predictions based on input features.
- **Request**:
  - **Headers**: `Content-Type: application/json`
  - **Body** (example):
    ```json
    {
        "features": [12.5, 180.3, 45.7, 12.8, 1500, 25.6]
    }
    ```
- **Response**:
  ```json
  {
      "Downtime": "Yes"
  }
  ```

## Testing the API

You can test the API using tools like **Postman** or **cURL**.

### Using Postman

#### Upload Endpoint
1. Set the method to `POST`.
2. URL: `http://127.0.0.1:5000/upload`
3. Go to the **Body** tab, select `form-data`, and upload your CSV file with the key name `file`.
4. Click **Send**.

#### Train Endpoint
1. Set the method to `POST`.
2. URL: `http://127.0.0.1:5000/train`
3. Click **Send**.

#### Predict Endpoint
1. Set the method to `POST`.
2. URL: `http://127.0.0.1:5000/predict`
3. Go to the **Body** tab, select `raw`, and paste the JSON input:
   ```json
   {
       "features": [12.5, 180.3, 45.7, 12.8, 1500, 25.6]
   }
   ```
4. Click **Send**.

## Notes
- Ensure the dataset uploaded includes the required features: `Torque(Nm)`, `Hydraulic_Pressure(bar)`, `Cutting(kN)`, `Coolant_Pressure(bar)`, `Spindle_Speed(RPM)`, `Coolant_Temperature`, and `Downtime` (target column).
- Update the paths for the model and preprocessing pipeline files if necessary.
- Check the Flask server logs for debugging in case of errors.

