import cv2
import threading
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN

# Load the model and processor
model_dir = "/home/alperenyildirim/Desktop/saved_model"
model = AutoModelForImageClassification.from_pretrained(model_dir)
image_processor = AutoImageProcessor.from_pretrained(model_dir)

# Create a pipeline
pipe = pipeline("image-classification", model=model, feature_extractor=image_processor, device=0 if torch.cuda.is_available() else -1)

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Function to preprocess the frame
def preprocess_frame(face):
    face_resized = cv2.resize(face, (224, 224))  # Resize the face for faster processing
    image = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
    return image

# Thread class for capturing frames
class VideoCaptureThread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self) 
        self.name = name
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# Thread class for processing frames
class VideoProcessingThread(threading.Thread):
    def __init__(self, name, capture_thread):
        threading.Thread.__init__(self)
        self.name = name
        self.capture_thread = capture_thread
        self.stopped = False

    def run(self):
        while not self.stopped:
            if self.capture_thread.frame is not None:
                frame = self.capture_thread.frame
                small_frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(rgb_frame)
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        face = frame[y1:y2, x1:x2]
                        image = preprocess_frame(face)
                        results = pipe(image)
                        label = results[0]['label']
                        confidence = results[0]['score']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.imshow('Emotion Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stopped = True

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()

# Create and start video capture thread
capture_thread = VideoCaptureThread("VideoCaptureThread")
capture_thread.start()

# Create and start video processing thread
processing_thread = VideoProcessingThread("VideoProcessingThread", capture_thread)
processing_thread.start()

# Wait for threads to complete
capture_thread.join()
processing_thread.join()
