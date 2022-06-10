from flask import Flask, render_template, request, send_from_directory, jsonify
import numpy as np
import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import label_map_util



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'

model = hub.load('my_ssd_mobnet/export/saved_model').signatures['serving_default']

def predict_label(img_path):
    category_index = label_map_util.create_category_index_from_labelmap('labelmap.pbtxt')
    
    img = cv2.imread(img_path)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.experimental.numpy.uint8)
    detections = model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    label = category_index[detections['detection_classes'][0]]['name']
   
    return label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
          
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

   
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if request.method == 'POST':    
         # Catch the image file from a POST request
        if 'file' not in request.files:
            return "Please try again. The Image doesn't exist"
        
        file = request.files.get('file')

        if not file:
            return
        if request.files:
            image = request.files['file']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            # image.save(img_path)
            # prediction = predict_label(img_path)
            prediction = predict_label(img_path)
            dicti = { 
                    "predict" : prediction
                    }
            
            return jsonify(dicti)
    # Return on a JSON format
    return jsonify(prediction="empty")

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

