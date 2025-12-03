

# ✅ **IMPORTS**

```python
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
```

### Meaning:

* `load_model`: load your trained emotion CNN
* `img_to_array`: convert image to Keras-friendly array
* `cv2`: webcam, image processing
* `numpy`: numerical operations

---

# ✅ **Load the face detector**

```python
face_classifier = cv2.CascadeClassifier(r'C:\Users\AHILI\Downloads\haarcascade_frontalface_default.xml')
```

Loads Haar Cascade XML file for **face detection**.

---

# ✅ **Load the trained emotion model**

```python
classifier = load_model(r'C:\Users\AHILI\Downloads\model.h5', compile=False)
```

* Loads your CNN (trained on FER images)
* `compile=False` avoids warnings when loading optimizers

---

# ✅ **Class names**

```python
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
```

These must match your training order.

---

# ✅ **Start webcam**

```python
cap = cv2.VideoCapture(0)
```

Opens your default webcam (camera index 0).

---

# 🚀 **MAIN LOOP — runs continuously**

```python
while True:
```

Captures frames, detects faces, predicts emotion.

---

# ✅ **Capture frame**

```python
_, frame = cap.read()
```

`frame` = image from webcam.

---

# ✅ **Convert to grayscale**

```python
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
```

Why grayscale?

* Face detection works best on grayscale
* Your model also expects grayscale (48×48×1)

---

# ✅ **Detect faces**

```python
faces = face_classifier.detectMultiScale(gray)
```

Returns list of detected face coordinates as `(x, y, w, h)`.

---

# ➿ **Loop over all faces found**

```python
for (x,y,w,h) in faces:
```

---

# 🟦 **Draw rectangle around face**

```python
cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
```

---

# 👁 **Extract face region (ROI)**

```python
roi_gray = gray[y:y+h, x:x+w]
roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
```

* Crop the face
* Resize to **48×48** → required by your CNN

---

# 🧼 **Preprocess the face image**

```python
roi = roi_gray.astype('float')/255.0
roi = img_to_array(roi)
roi = np.expand_dims(roi,axis=0)
```

Steps:

1. Convert to float
2. Normalize to 0–1
3. Convert to array
4. Add batch dimension → shape becomes `(1, 48, 48, 1)`

This is exactly what your model expects.

---

# 🔮 **Predict emotion**

```python
prediction = classifier.predict(roi)[0]
```

`prediction` = array of probabilities, like:

```
[0.10, 0.05, 0.03, 0.70, 0.05, 0.02, 0.05]
```

---

# 🏷 **Get label**

```python
label = emotion_labels[prediction.argmax()]
```

`argmax()` → index of highest value
Example: if index = 3 → 'Happy'

---

# 📝 **Display label on screen**

```python
cv2.putText(frame, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
```

---

# 🎥 **Show video window**

```python
cv2.imshow('Emotion Detector', frame)
```

---

# ⏹ **Quit when 'q' is pressed**

```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

---

# 🔚 **Release resources**

```python
cap.release()
cv2.destroyAllWindows()
```

---

# ⭐ **FINAL SUMMARY**

Your program does:

1. Start webcam
2. Detect face using Haar Cascade
3. Extract face region
4. Convert to grayscale + resize to 48×48
5. Normalize + reshape
6. Feed into your CNN
7. Get emotion prediction
8. Draw label on screen
9. Run continuously until 'q' is pressed

This is a **full real-time facial emotion recognition system**.

---


