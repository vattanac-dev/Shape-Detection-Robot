import cv2
import numpy as np
import tensorflow as tf
import json
import time
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "shape_mobilenet.h5")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")


with open(CLASS_INDICES_PATH, "r") as f:
    # e.g. {"circle": 0, "square": 1, "triangle": 2}
    class_indices = json.load(f)
    # {0: "circle", 1: "square", 2: "triangle"}
    classes = {v: k for k, v in class_indices.items()}


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")


INPUT_SIZE = 320


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_time = time.time()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: failed to read frame from camera.")
        continue

    current_time = time.time()
    fps = 1.0 / (current_time -
                 prev_time) if current_time != prev_time else 0.0
    prev_time = current_time

    h, w = frame.shape[:2]

    box_size = int(min(w, h) * 0.5)
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2].copy()

    if roi.size == 0:
        continue

    # ---- Preprocess for model ----
    img = cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ---- Predict ----
    preds = model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])
    label = classes.get(class_id, "unknown")

    color = (0, 255, 0)  # green
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} ({confidence:.2f})"
    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Hold shape inside the box",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Webcam Shape Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
