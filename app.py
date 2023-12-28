from flask import Flask, request, jsonify
import pickle
import numpy as np
import sklearn


model = pickle.load(open("rf_model.pkl", 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return "Happy"


@app.route('/predict', methods=['POST'])
def predict():
    N = request.form.get('N')
    P = request.form.get('P')
    K = request.form.get('K')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')
    input_query = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    result = model.predict(input_query)[0]
    return jsonify({'Crop': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
