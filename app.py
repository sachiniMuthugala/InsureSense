from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('heart_disease_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('homePage.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = request.form['gender']
    weight = float(request.form['weight'])
    systolic_bp = int(request.form['systolic_bp'])
    diastolic_bp = int(request.form['diastolic_bp'])
    cholesterol = int(request.form['cholesterol'])
    gender_male = 0
    gender_female = 0
    if gender == 'male':
        gender_male = 1
    else:
        gender_female = 1
    data = pd.DataFrame({'age': [age],
                         'gender_male': [gender_male],
                         'gender_female': [gender_female],
                         'weight': [weight],
                         'systolic_bp': [systolic_bp],
                         'diastolic_bp': [diastolic_bp],
                         'cholesterol': [cholesterol]})
    prediction = model.predict_proba(data)[0][1]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
