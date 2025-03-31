import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import tempfile
import os
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained TensorFlow model
model = tf.keras.models.load_model("vegetable_classifier.h5")
img_size = 128  # Resize images for model input
categories = ["Fresh", "Spoiled"]  # Update based on your trained model categories

# Initial inventory
inventory = [
    {"id": 1, "name": "Chicken", "quantity": 20, "expires": "2025-04-05", "visionDetected": "Fresh"},
    {"id": 2, "name": "Tomatoes", "quantity": 0, "expires": "2025-04-02", "visionDetected": "Not detected"},
    {"id": 3, "name": "Bread", "quantity": 10, "expires": "2025-03-31", "visionDetected": "Near-expiry"},
]


def classify_frame(frame):
    """Classify a single image frame using the TensorFlow model."""
    img = cv2.resize(frame, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = categories[class_index]
    confidence = prediction[0][class_index]

    return f"{class_label} (Confidence: {confidence:.2f})"


@app.post("/api/upload_video")
async def upload_video(video: UploadFile = File(...)):
    """Receives a video file, extracts frames, classifies them, and returns results."""
    try:
        # Save the video file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
            temp_video.write(video.file.read())
            temp_video_path = temp_video.name

        # Open the video file with OpenCV
        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "Could not open video file"})

        classification_results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Stop if there are no more frames

            if frame_count % 10 == 0:  # Process every 10th frame to reduce processing load
                result = classify_frame(frame)
                classification_results.append(result)

            frame_count += 1

        cap.release()
        os.remove(temp_video_path)  # Clean up temp file

        if not classification_results:
            return {"classification": "No classification results available"}

        # Get the most frequent classification
        most_common_class = Counter(classification_results).most_common(1)[0][0]

        return {"classification": most_common_class}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/inventory")
async def get_inventory():
    """Return the current inventory."""
    return inventory


if __name__ == "__main__":
    print("Starting FastAPI server with TensorFlow model integration...")
    uvicorn.run(app, host="0.0.0.0", port=5000)