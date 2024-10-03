from django.http import StreamingHttpResponse
from ultralytics import YOLO
import cv2
from django.shortcuts import render
from django.conf import settings
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models  # <-- Add this import for ResNet models
from torchvision.models import ResNet18_Weights
from PIL import Image

base_dir = Path(settings.BASE_DIR)
last_model_path = base_dir / 'Nari_Shakti' / 'models' / 'last.pt'
last1_model_path = base_dir / 'Nari_Shakti' / 'models' / 'last1.pt'
emotion_model_path = base_dir / 'Nari_Shakti' / 'models' / 'emotion_detection.pth'

gender_classifier = YOLO(last1_model_path)
face_detector = YOLO(last_model_path)

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassifier, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)  
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  

    def forward(self, x):
        return self.resnet(x)

emotion_classifier = EmotionClassifier(num_classes=7)
emotion_classifier.load_state_dict(torch.load(emotion_model_path, map_location=torch.device('cpu')), strict=False)
emotion_classifier.eval()

emotion_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

emotion_labels = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}

def process_frame(frame):

    male_count = 0
    female_count = 0
    sos_triggered = False

    results = face_detector(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int) 

            face_img = frame[y1:y2, x1:x2]

            face_img_resized = cv2.resize(face_img, (224, 224))

            gender_results = gender_classifier(face_img_resized)

            if gender_results and len(gender_results) > 0:
                predicted_class = gender_results[0].probs.top1
                confidence = gender_results[0].probs.top1conf
                gender_label = gender_results[0].names[predicted_class]

                if gender_label == 'male':
                    male_count += 1
                elif gender_label == 'female':
                    female_count += 1
            else:
                gender_label = 'Unknown'

            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img_resized_gray = cv2.resize(face_img_gray, (48, 48))
            face_img_pil = Image.fromarray(face_img_resized_gray)

            face_tensor = emotion_transform(face_img_pil).unsqueeze(0) 

            with torch.no_grad():
                emotion_output = emotion_classifier(face_tensor)
                emotion_prediction = torch.argmax(emotion_output, dim=1).item()
                emotion_label = emotion_labels[emotion_prediction]

            if gender_label == 'female' and emotion_label in ['Disgusted', 'Angry', 'Surprised', 'Fearful']:
                if male_count > female_count:
                    sos_triggered = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{gender_label} ({confidence:.2f})", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {emotion_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"Male: {male_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if sos_triggered:
        trigger_sos()

    return frame

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def trigger_sos():

    sender_email = "subhrajyoti479@gmail.com"  
    sender_password = "joox gtjr zbax puci"  
    recipient_email = "subhrajyotibasu0@gmail.com"  

    subject = "SOS Alert: Potential Threat Detected!"
    body = "Alert: A potential threat has been detected!"
    
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  
        server.login(sender_email, sender_password) 

        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)

        print("SOS email sent successfully!")

        server.quit()
    except Exception as e:
        print(f"Failed to send SOS email: {e}")


def generate_video_stream():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)

        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

def live_feed(request):
    return StreamingHttpResponse(generate_video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def display_live_feed(request):
    return render(request, 'live_feed.html')
