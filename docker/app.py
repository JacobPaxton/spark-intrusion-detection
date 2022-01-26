# Import all the packages you need for your model below
import pickle
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier

# Import Flask for creating API
from flask import Flask, request
port = int(os.environ.get('PORT', 5000))

# Load the trained model from current directory
with open('./anomaly_detection_model.sav', 'rb') as model_sav:
    rf = pickle.load(model_sav)

# Load the trained scaler from current directory
with open('./anomaly_scaler.sav', 'rb') as scaler_sav:
    scaler = pickle.load(scaler_sav)

# Initialise a Flask app
app = Flask(__name__)

# Create an API endpoint
@app.route('/predict')
def predict_anomaly():
    # read all necessary request parameters
    srv_count = request.args.get('srv_count')
    num_failed_logins = request.args.get('num_failed_logins')
    # create numpy array for inputs
    input_array = np.array([[srv_count, num_failed_logins]])
    # scale the input
    scaled_inputs = scaler.transform(input_array)
    # predict the scaled input
    predict_result = rf.predict(scaled_inputs)
    # return the result back
    return 'Predicted result for observation ' + str(input_array) + ' is: ' + str(predict_result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=port)