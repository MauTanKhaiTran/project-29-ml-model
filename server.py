from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openai
openai.api_key = "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"
import joblib
import mahotas
from sklearn.preprocessing import MinMaxScaler


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

# classes for CNN
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

# classes for other models
leaf_classes_2 = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'Healthy']

# Converting each image to RGB from BGR format
def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img

# Conversion to HSV image format from RGB
def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img

# image segmentation
# for extraction of green and brown color
def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    bins = 8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

#  pre-process image for dt and rf model
def preprocess_image(img):    # Running Function Bit By Bit
    RGB_BGR       = rgb_bgr(img)
    BGR_HSV       = bgr_hsv(RGB_BGR)
    IMG_SEGMENT   = img_segmentation(RGB_BGR,BGR_HSV)
    # Call for Global Fetaure Descriptors
    fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
    fv_haralick   = fd_haralick(IMG_SEGMENT)
    fv_histogram  = fd_histogram(IMG_SEGMENT)
    # Concatenate 
    processed_img = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    return processed_img


def model_predict(img_path, modelType):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    if modelType == "CNN":
        cnn_model = load_model('all_models/model.h5', compile=False)
        cnn_model.compile()
        data = np.array([img])
        result = cnn_model.predict(data)[0]
        predicted = result.argmax()
        pred_answer = leaf_classes[predicted]

        # To get confidence scores for each class with class labels:
        class_confidence_scores = dict(zip(leaf_classes, result))
        # Extract class labels and confidence scores
        class_labels = list(class_confidence_scores.keys())
        confidence_scores = list(class_confidence_scores.values())
    elif modelType == "Random Forest":
        rf_model = joblib.load('all_models/rf_model.pkl')
        img = preprocess_image(img)
        # Predict using the random forest classifier
        pred_answer = rf_model.predict([img])[0]
        pred_answer = pred_answer.split("___")[1]

        confidence_scores = rf_model.predict_proba([img])[0]
    elif modelType == "Decision Tree":
        dt_model = joblib.load('all_models/dt_model.pkl')
        img = preprocess_image(img)
        # Predict using the decision tree classifier
        pred_answer = dt_model.predict([img])[0]
        pred_answer = pred_answer.split("___")[1]

        confidence_scores = dt_model.predict_proba([img])[0]
    elif modelType == "Linear Discriminant Analysis":
        lda_model = joblib.load('all_models/lda_model.pkl')
        img = preprocess_image(img)
        # Predict using the lda classifier
        pred_answer = lda_model.predict([img])[0]
        pred_answer = pred_answer.split("___")[1]

        confidence_scores = lda_model.predict_proba([img])[0]
    elif modelType == "Logistic Regression":
        lr_model = joblib.load('all_models/lr_model.pkl')

        img = preprocess_image(img)
        # Create a MinMaxScaler instance
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Scale the preprocessed image
        scaled_img = scaler.fit_transform([img])        # Predict using the logistic regression classifier
        pred_answer = lr_model.predict(scaled_img)[0]
        pred_answer = pred_answer.split("___")[1]

        confidence_scores = lr_model.predict_proba(scaled_img)[0]

    # Calculate percentages
    total_confidence = sum(confidence_scores)
    percentages = [score / total_confidence * 100 for score in confidence_scores]

    # # Create a bar plot
    plt.figure(figsize=(10, 5))
    if modelType == "CNN":
        bars = plt.barh(class_labels, confidence_scores, color='royalblue')
    else:
        bars = plt.barh(leaf_classes_2, confidence_scores, color='royalblue')

    # # Add percentages as text on the bars
    # for bar, percent in zip(bars, percentages):
    #     plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f'{percent:.2f}%', va='center')

    plt.xlabel('Confidence Score')  # Update the x-axis label
    plt.title(f'{modelType} Prediction Confidence Scores')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the highest confidence at the top

    # Save the plot to a file (e.g., PNG format)
    plot_filename = 'plot.png'
    plt.savefig(plot_filename)

    pred_answer = pred_answer.replace("_", " ")
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        imageType = request.form['imageType']
        modelType = request.form['modelType']
        print("Image type:", imageType)
        print("Model type:", modelType)

        # Get the image file from the request
        image = request.files['image']

        # Save the image file
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)

        # Perform prediction and return response
        result = model_predict(file_path, modelType)
        solution = generate_chat_response(result)

        # print(solution)
        response_data = {'prediction': result, 'solution': solution}
        return jsonify(response_data)
    return None

@app.route('/get_plot', methods=['GET'])
def get_plot():
    # Make sure to provide the correct path to the saved plot file
    plot_filename = 'plot.png'

    # Check if the plot file exists
    if os.path.exists(plot_filename):
        print("Plot sent")
        return send_file(plot_filename, mimetype='image/png')
    else:
        print("Plot not found")
        return jsonify({'error': 'Plot not found'})
    
@app.route('/get_training_history', methods=['GET'])
def get_history():
    # Make sure to provide the correct path to the saved plot file
    plot_filename = 'training_plot.png'

    # Check if the plot file exists
    if os.path.exists(plot_filename):
        print("Training history sent")
        return send_file(plot_filename, mimetype='image/png')
    else:
        print("Training history not found")
        return jsonify({'error': 'Training history not found'})

if __name__ == '__main__':
    app.run(debug=True)