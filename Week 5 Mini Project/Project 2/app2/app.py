import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import base64

app = Flask(__name__)

# Load your saved model
model = tf.keras.models.load_model('app2\my_model.keras')

# Preprocess function to handle the CSV and extract the hourly frequency
def preprocess_data(df):
    # Extract the hour from 'created_at' and count frequencies
    df['hour'] = pd.to_datetime(df['created_at']).dt.hour
    hourly_freq = df['hour'].value_counts().sort_index()
    
    # Ensure all hours are represented (even those with 0 tweets)
    hourly_freq = hourly_freq.reindex(np.arange(24), fill_value=0)
    
    # Convert to the format your model expects
    X = np.array(hourly_freq).reshape(1, -1)  # Adjust as per your model's input shape
    
    return X, hourly_freq

# Route for handling the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['csv-file']
    
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    df = pd.read_csv(file)
    
    # Preprocess the data
    X, hourly_freq = preprocess_data(df)
    
    # Predict using the model
    prediction = model.predict(X)
    
    # Plotting the intensity of the traffic
    plt.figure(figsize=(10, 5))
    plt.plot(hourly_freq.index, hourly_freq.values, label='Tweet Frequency')
    plt.title('Hourly Tweet Frequency')
    plt.xlabel('Hour')
    plt.ylabel('Number of Tweets')
    plt.legend()
    plt.grid(True)
    
    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return jsonify({'prediction': prediction.tolist(), 'plot': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
