from flask import Flask, request, redirect, render_template, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import cv2
import numpy as np
import openai
openai.api_key = "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"

app = Flask(__name__)
CORS(app)

# Load saved model
# model = tf.keras.models.load_model('my_model_2')
# model._make_predict_function()          # Necessary

# def model_predict(img_path, model):
#     image = load_img(img_path, target_size=(224, 224))
#     image_array = img_to_array(image)
#     image_array = np.expand_dims(image_array, axis=0)

#     preds = model.predict(image_array)
#     if preds[0] < 0.3:
#         return "DISEASED"
#     else:
#         return "HEALTHY"

leaf_classes = [
    'Target_Spot',
    'Late_blight',
    'Mosaic_virus',
    'Leaf_Mold',
    'Bacterial_spot',
    'Early_blight',
    'Healthy',
    'Yellow_Leaf_Curl_Virus',
    'Two-spotted_spider_mite',
    'Septoria_leaf_spot'
]

model = load_model('model/model.h5')
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    data = np.array([img])
    result = model.predict(data)[0]
    predicted = result.argmax()
    pred_answer = leaf_classes[predicted]
    return pred_answer


def generate_chat_response(prompt):
    if prompt != "Healthy":
        prompts = [{"role": "system", "content": "You are a plant disease expert. You provide the farmer with the solution to their problem."},
                                {"role": "user", "content": f"My crops are having {prompt}. Please provide me a short solution."}]
        
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompts)
        # return "hello world"
        return response["choices"][0]["message"]["content"]
    else:
        return "Your plant is healthy. No need to worry."

@app.route('/', methods=['GET'])
def index():
    return "hello"

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        # Get text data from form fields
        firstName = request.form['firstName']
        lastName = request.form['lastName']
        state = request.form['state']
        country = request.form['country']
        postalCode = request.form['postalCode']

        # Get the image file from the request
        image = request.files['image']

        # Save the image file
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)

        # Perform prediction and return response
        result = model_predict(file_path, model)
        solution = generate_chat_response(result)

        print(solution)
        response_data = {'prediction': result, 'solution': solution}
        return jsonify(response_data)
    return None

if __name__ == '__main__':
    app.run(debug=True)