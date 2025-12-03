from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import sounddevice as sd
import librosa

face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

image_classifier = load_model(r"image_detection.h5", compile = False)
audio_classifier = load_model(r"audio_emotion_model.h5", compile = False)

image_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
audio_labels = ["fear", "angry", "disgust", "neutral", "sad", "pleasantly surprised", "happy"]

sr = 22050
duration = 2

def predict_audio_emotion():
    audio = sd.rec(int(duration * sr), samplerate = sr, channels = 1, dtype = 'float32')
    sd.wait()
    audio = audio.flatten()

    mfcc = librosa.feature.mfcc(y = audio, sr = sr, n_mfcc = 40)
    mfcc_mean = np.mean(mfcc.T, axis = 0)

    mfcc_mean = mfcc_mean.reshape(40, 1)
    mfcc_mean = np.expand_dims(mfcc_mean, axis = 0)

    preds = audio_classifier.predict(mfcc_mean)[0]
    audio_emotion = audio_labels[preds.argmax()]
    return audio_emotion

cap = cv2.VideoCapture(0)

audio_timer = 0
audio_interval = 50
last_audio_emotion = ""


while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    #periodic audio prediction
    if audio_timer % audio_interval == 0:
        last_audio_emotion = predict_audio_emotion()
    audio_timer += 1

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = image_classifier.predict(roi)[0]
            image_emotion = image_labels[prediction.argmax()]

            cv2.putText(frame, "Face: " + image_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.putText(frame, "Audio: " + last_audio_emotion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Emotion Detector (Video and Audio)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()