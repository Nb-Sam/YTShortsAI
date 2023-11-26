# app/routes.py
from flask import render_template, request, jsonify
from app import app
import os
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from pytube import YouTube

model = InceptionV3(weights='imagenet')

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    results = []
    for _, label, score in decoded_predictions:
        results.append({'label': label, 'score': round(float(score), 2)})

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        youtube_url = request.form['youtube_url']

        # Download YouTube video
        video = YouTube(youtube_url)
        video_stream = video.streams.filter(file_extension='mp4').first()
        video_path = os.path.join('downloads', 'input_video.mp4')
        video_stream.download('downloads', 'input_video')

        # Extract short clip (e.g., first 10 seconds)
        output_path = os.path.join('downloads', 'output_video.mp4')
        os.system(f'ffmpeg -i {video_path} -ss 00:00:00 -t 00:00:10 -c copy {output_path}')

        # Process frames
        frames_folder = os.path.join('downloads', 'frames')
        os.makedirs(frames_folder, exist_ok=True)
        cap = cv2.VideoCapture(output_path)
        success, frame = cap.read()
        count = 0

        while success:
            frame_path = os.path.join(frames_folder, f'frame{count}.jpg')
            cv2.imwrite(frame_path, frame)

            classify_result = classify_image(frame_path)
            print(f"Frame {count + 1} Classification: {classify_result}")

            success, frame = cap.read()
            count += 1

        cap.release()

        return jsonify({'success': True, 'message': 'Video processed successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
