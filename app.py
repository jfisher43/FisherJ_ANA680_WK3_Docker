from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# Initialize the flask app
app = Flask(__name__)

# Load the model
filename = 'knn_model.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, density, pH, sulphates, alcohol]])
        print("Input data for prediction:", input_data)

        pred = model.predict(input_data)
        print("Prediction result:", pred)

        return render_template('index.html', predict=pred[0])
    except Exception as e:
        print("Error occurred:", e)
        return render_template('index.html', predict="An error occurred. Please try again.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)