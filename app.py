from flask import Flask, render_template, request
import pandas as pd
import joblib
import sqlite3
import os

app = Flask(__name__)

# Ensure DB is initialized
def init_db():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT,
            email TEXT,
            age INTEGER,
            gender TEXT,
            occupation TEXT,
            sleep_hours INTEGER,
            exercise_hours INTEGER,
            phone_usage INTEGER,
            work_pressure INTEGER,
            mood_swings TEXT,
            meal_timing TEXT,
            predicted_stress_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load model and encoders
    model = joblib.load('stress_model.pkl')
    gender_encoder = joblib.load('gender_encoder.pkl')
    occupation_encoder = joblib.load('occupation_encoder.pkl')
    mind_swing_encoder = joblib.load('mind_swing_encoder.pkl')
    meal_timing_encoder = joblib.load('meal_timing_encoder.pkl')
    stress_encoder = joblib.load('stress_encoder.pkl')

    # Form data
    data = request.form
    name = data['Name']
    phone = data['Mobile']
    email = data['Email']
    age = int(data['Age'])
    gender = data['Gender']
    occupation = data['Occupation']
    sleep_hours = int(data['SleepHours'])
    exercise_hours = int(data['ExerciseHours'])
    work_pressure = int(data['WorkPressure'])
    phone_usage = int(data['PhoneUsageHours'])
    mind_swing = data['MoodSwings']
    meal_timing = data['MealsOnTime']

    # Prediction input
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'Sleep_Hours': sleep_hours,
        'Exercise_Hours': exercise_hours,
        'Work_Pressure': work_pressure,
        'Phone_Usage': phone_usage,
        'Mind_Swing': mind_swing,
        'Meal_Timing': meal_timing
    }])

    input_df['Gender'] = gender_encoder.transform(input_df['Gender'])
    input_df['Occupation'] = occupation_encoder.transform(input_df['Occupation'])
    input_df['Mind_Swing'] = mind_swing_encoder.transform(input_df['Mind_Swing'])
    input_df['Meal_Timing'] = meal_timing_encoder.transform(input_df['Meal_Timing'])

    # Predict
    prediction = model.predict(input_df)
    predicted_label = stress_encoder.inverse_transform(prediction)[0]

    # Save to SQLite DB
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_data (
            name, phone, email, age, gender, occupation,
            sleep_hours, exercise_hours, phone_usage,
            work_pressure, mood_swings, meal_timing, predicted_stress_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        name, phone, email, age, gender, occupation,
        sleep_hours, exercise_hours, phone_usage,
        work_pressure, mind_swing, meal_timing, predicted_label
    ))
    conn.commit()
    conn.close()

    return render_template('result.html', stress_level=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
