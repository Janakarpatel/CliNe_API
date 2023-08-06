from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import pickle5 as pickle

app = Flask(__name__)
CORS(app)

# Load the saved model
model = tf.keras.models.load_model('models/model8_10.h5',compile=False)

# Load the saved weights
model.load_weights('models/weights_10.h5')

# Load the saved tokenizer
with open('models/tokenizer_10.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

@app.route('/')
def hello():
    return "Welcome to our CliNe's authorized API! To access our services through the Chrome extension, simply follow on web store to integrate it seamlessly with your browser. We value intellectual property rights, and all data and content provided through our API are protected by copyright laws. By accessing and using our API, you agree to abide by the copyright terms and conditions. Thank you for choosing our service, Happy browsing! © [CliNe.AI] 2023. All rights reserved."

# Define the API route
@app.route('/predict', methods=['POST'])
def predict():
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
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run()
