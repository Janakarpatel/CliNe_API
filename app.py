from flask import Flask, jsonify, request,render_template
from flask_cors import CORS
import tensorflow as tf
import pickle5 as pickle
import os
import logging

#configuration of log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

port = os.environ.get('PORT',10000)
PORT = 10000

# Load the saved model
model = tf.keras.models.load_model('models/model8_10.h5',compile=False)

# Load the saved weights
model.load_weights('models/weights_10.h5')

# Load the saved tokenizer
with open('models/tokenizer_10.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

app = Flask(__name__)
# CORS(app)


# @app.route('/')
# def index():
#     return "Welcome to our CliNe's authorized API! To access our services through the Chrome extension, simply follow on web store to integrate it seamlessly with your browser. We value intellectual property rights, and all data and content provided through our API are protected by copyright laws. By accessing and using our API, you agree to abide by the copyright terms and conditions. Thank you for choosing our service, Happy browsing! © [CliNe.AI] 2023. All rights reserved."

# Define the API route
@app.route('/', methods=['POST'])
def predict():
    try:
        # Get the text data from the request body
        text = request.json['text']
    
        # Convert the text to a sequence using the loaded tokenizer
        sequence = tokenizer.texts_to_sequences([text])
    
        # Pad the sequence to a fixed length
        max_len = 20 # or whatever your sequence length is
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len, padding='post')
    
        # Use the loaded model to make a prediction
        prediction = model.predict(padded)
    
        # Return the predicted label as a JSON response
        label = 'clickbait' if prediction[0][0] > 0.5 else 'not clickbait'
        logging.info(f"Input text: {text}")
        logging.info(f"Prediction: {label}")
    
        return jsonify({'label': label})
    
    except Exception as e:
        # Log any errors that occur during prediction
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=port)
