from django.http import StreamingHttpResponse
from ultralytics import YOLO
import cv2
from django.shortcuts import render
import os
from django.conf import settings
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models  # <-- Add this import for ResNet models
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np

# Using pathlib for path construction
base_dir = Path(settings.BASE_DIR)
last_model_path = base_dir / 'Nari_Shakti' / 'models' / 'last.pt'
last1_model_path = base_dir / 'Nari_Shakti' / 'models' / 'last1.pt'
emotion_model_path = base_dir / 'Nari_Shakti' / 'models' / 'emotion_detection.pth'

# Load pre-trained models
gender_classifier = YOLO(last1_model_path)
face_detector = YOLO(last_model_path)

# Load the emotion detection model (ResNet-based)
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassifier, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Initialize ResNet18 without pre-training
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify to accept grayscale input
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # Adjust the final layer for 7 emotion classes

    def forward(self, x):
        return self.resnet(x)

# Load the emotion detection model
emotion_classifier = EmotionClassifier(num_classes=7)
emotion_classifier.load_state_dict(torch.load(emotion_model_path, map_location=torch.device('cpu')), strict=False)
emotion_classifier.eval()

# Transformation for emotion detection (48x48 grayscale images)
emotion_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])

# Dictionary to map emotion class indices to labels
emotion_labels = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}

def process_frame(frame):
    # Initialize gender and emotion counts
    male_count = 0
    female_count = 0

    # Face detection
    results = face_detector(frame)

    # Process the results and get bounding boxes
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Bounding box coordinates

            # Crop the face region
            face_img = frame[y1:y2, x1:x2]

            # Resize the face image to fit the gender classifier input size (224x224)
            face_img_resized = cv2.resize(face_img, (224, 224))

            # Gender classification
            gender_results = gender_classifier(face_img_resized)

            # Get the predicted class
            if gender_results and len(gender_results) > 0:
                predicted_class = gender_results[0].probs.top1
                confidence = gender_results[0].probs.top1conf
                gender_label = gender_results[0].names[predicted_class]

                # Increment gender count based on prediction
                if gender_label == 'male':
                    male_count += 1
                elif gender_label == 'female':
                    female_count += 1
            else:
                gender_label = 'Unknown'

            # Emotion detection
            # Convert the face image to grayscale, resize to 48x48, and normalize
            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img_resized_gray = cv2.resize(face_img_gray, (48, 48))
            face_img_pil = Image.fromarray(face_img_resized_gray)

            # Now apply the transform
            face_tensor = emotion_transform(face_img_pil).unsqueeze(0)  # Add batch dimension
            #face_tensor = emotion_transform(face_img_resized_gray).unsqueeze(0)  # Add batch dimension

            # Predict emotion
            with torch.no_grad():
                emotion_output = emotion_classifier(face_tensor)
                emotion_prediction = torch.argmax(emotion_output, dim=1).item()
                emotion_label = emotion_labels[emotion_prediction]

            # Draw bounding box, gender, and emotion label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{gender_label} ({confidence:.2f})", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {emotion_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display male and female counts on the frame
    cv2.putText(frame, f"Male: {male_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def generate_video_stream():
    # Replace '0' with the URL or ID of the CCTV camera stream
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        frame = process_frame(frame)

        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Yield the frame to the response stream
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

def live_feed(request):
    # Streaming HTTP response for live video feed
    return StreamingHttpResponse(generate_video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def display_live_feed(request):
    return render(request, 'live_feed.html')



