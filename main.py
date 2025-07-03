from retinaface import RetinaFace
import cv2
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
from train import train_gender_classifier
from train2 import train_facenet
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
import os
import csv

# Define dataset paths
train_dir = r'E:\comsys hackathon\Comys_Hackathon5\Task_A\train'
val_dir = r'E:\comsys hackathon\Comys_Hackathon5\Task_A\val'
facenet_train_dir = r"E:\comsys hackathon\Comys_Hackathon5\Task_B\train"

# Train or load the gender classification model
try:
    model = load_model('gender_classification_efficientnet_final.keras')
    print("Loaded trained model from 'gender_classification_efficientnet.keras'.")
except (IOError, OSError):
    print("Trained model not found. Training the model now...")
    model, history = train_gender_classifier(train_dir, val_dir)
    print("Training completed. Model saved as 'gender_classification_efficientnet.keras'")

# FaceNet model loading or training
facenet_model_path = "facenet_triplet.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    facenet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
    # Only modify these layers if it's a new model
    if not os.path.exists(facenet_model_path):
        facenet.last_linear = torch.nn.Linear(facenet.last_linear.in_features, 128)
        facenet.last_bn = torch.nn.BatchNorm1d(128)
    else:
        facenet.load_state_dict(torch.load(facenet_model_path, map_location=device))
    print(f"Loaded FaceNet model from '{facenet_model_path}'.")
except Exception as e:
    print(f"FaceNet model error: {e}. Training the model now...")
    train_facenet(facenet_train_dir)
    facenet = InceptionResnetV1(pretrained='vggface2', classify=False)
    facenet.last_linear = torch.nn.Linear(facenet.last_linear.in_features, 128)
    facenet.last_bn = torch.nn.BatchNorm1d(128)
    facenet.load_state_dict(torch.load(facenet_model_path, map_location=device))
    facenet = facenet.eval().to(device)

# -------------------- VIDEO PROCESSING BEGINS --------------------

# Setup video input/output
video_path = "samplevideo.mp4"  # 0 for webcam, or path to video file e.g. 'input.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video source: {video_path}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Optional: write predictions to CSV
csv_file = open('predictions.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Face_ID', 'Gender', 'Probability'])

frame_count = 0

try:
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_count += 1

        cv2.imwrite("temp_frame.jpg", img)
        faces = RetinaFace.detect_faces("temp_frame.jpg")
        
        if not faces or not isinstance(faces, dict):
            faces = {}
            out.write(img)
            continue

        for face_id, face_data in faces.items():
            try:
                facial_area = face_data['facial_area']
                x1, y1, x2, y2 = facial_area
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                face = img[y1:y2, x1:x2]
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_resized = face_pil.resize((128, 128))
                face_array = np.array(face_resized)

                # For gender classification
                face_preprocessed = preprocess_input(face_array.copy())
                face_preprocessed = np.expand_dims(face_preprocessed, axis=0)

                prediction = model.predict(face_preprocessed)
                predicted_class = np.argmax(prediction, axis=1)[0]
                class_label = 'Male' if predicted_class == 0 else 'Female'
                prob = prediction[0][predicted_class]

                label = f"{class_label} ({prob:.2f})"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Save prediction to CSV
                csv_writer.writerow([frame_count, face_id, class_label, round(prob, 2)])

                # For FaceNet embedding
                face_tensor = torch.tensor(face_array).float().permute(2, 0, 1).unsqueeze(0)
                face_tensor = (face_tensor - 127.5) / 128.0  # Normalization for FaceNet
                face_tensor = face_tensor.to(device)

                with torch.no_grad():
                    embedding = facenet(face_tensor)
                # Optionally save or use embedding here

            except Exception as e:
                print(f"Error processing face {face_id} in frame {frame_count}: {str(e)}")
                continue

        out.write(img)

        # Dynamically resize display
        height, width = img.shape[:2]
        scale = min(1200 / width, 800 / height)
        display_dim = (int(width * scale), int(height * scale))
        display_img = cv2.resize(img, display_dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('Detected Faces with RetinaFace (Video)', display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error during video processing: {str(e)}")

finally:
    # Cleanup
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")