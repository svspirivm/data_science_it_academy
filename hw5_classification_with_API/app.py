from flask import Flask, request, jsonify
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# creating a Flask application
app = Flask(__name__)

# Load the model
filename = 'scaler.pkl'

with open(filename,'rb') as f:
    loaded_scaler = pickle.load(f)

filename = 'extra_trees_classifier.pkl'

with open(filename,'rb') as f:
    loaded_model = pickle.load(f)

# creating target
target_class = np.array(['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II'], dtype=object)
le = LabelEncoder()
target_asced = le.fit_transform(target_class)

# creating predict url and only allowing post requests.
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Post request
    data = request.get_json()
    # converting a json request to the model format
    df = pd.read_json(data, orient='split')
    print('\n\n\n Request  \n', df.head())
    # Make prediction
    df_norm = pd.DataFrame(data=loaded_scaler.transform(df),
              columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])
    df_norm.drop('SMOKE', axis = 1, inplace = True)
    df_norm['BMI'] = df_norm['Weight'] / (df_norm['Height'] ** 2)
    pred = le.inverse_transform(loaded_model.predict(df_norm))
    print('\n\n\n Prediction   ', pred, '\n\n')
    # returning a prediction as json
    responses = pd.Series(pred).to_json(orient='values')
    return (responses)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)