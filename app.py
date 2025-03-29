from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Dummy user database
users = {'admin': 'password123'}  # Replace with a database for real authentication

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('predict'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['user'] = username  # Store user in session
            return redirect(url_for('predict'))
        else:
            return render_template('login.html', error="Invalid username or password")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user from session
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))  # Restrict access if not logged in

    if request.method == 'POST':
        try:
            # Get input data from form correctly
            data = CustomData(
                gender=request.form.get('gender'),
                logical_reasoning=request.form.get('reasoning'),  # Fixed key name
                learning_style=request.form.get('learning_style'),
                stress_level=request.form.get('stress level'),
                tutions=request.form.get('tutions'),
                writing_score=float(request.form.get('writing_score', 0)),  # Default to 0 if missing
                reading_score=float(request.form.get('reading_score', 0))
            )

            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:", pred_df)

            predict_pipeline = PredictPipeline()
            print("Running Prediction...")
            results = predict_pipeline.predict(pred_df)
            print("Prediction Complete:", results)

            return render_template('home.html', results=results)

        except Exception as e:
            print("Error during prediction:", str(e))
            return render_template('home.html', error="Prediction failed. Please check input values.")

    return render_template('home.html')  # Handle GET request properly    
if __name__ == '__main__':
    app.run(debug=True)
