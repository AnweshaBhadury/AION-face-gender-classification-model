# main.py – flexible pipeline for webcam, video, or single image
# ------------------------------------------------------------
# Usage examples:
#   python main.py --source 0              # webcam
#   python main.py --source myvideo.mp4    # video file
#   python main.py --source photo.jpg      # image file
#   python main.py                         # interactive prompt

import os
import csv
import argparse
import cv2
import torch
import numpy as np
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input

from train import train_gender_classifier
from train2 import train_facenet

# --------------------------- Helpers ---------------------------

def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}

# Adaptive resize for display (keeps width ≤ max_w)
def adaptive_resize(img: np.ndarray, max_w: int = 960) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

# --------------------------- CLI Args --------------------------

parser = argparse.ArgumentParser(description="Face‑Gender classification on webcam / video / image")
parser.add_argument("--source", help="0 (webcam), video file, or image file path")
args = parser.parse_args()

SOURCE = args.source.strip() if args.source else input("Enter source (0/webcam, video file, or image path): ").strip()

# ---------------------- Load Gender Model ----------------------
print("[INFO] Loading EfficientNet gender classifier …")
try:
    gender_model = load_model("gender_classification_efficientnet_final.keras")
except (IOError, OSError):
    DATA_ROOT = r"E:\comsys hackathon\Comys_Hackathon5"
    gender_model, _ = train_gender_classifier(
        os.path.join(DATA_ROOT, "Task_A", "train"),
        os.path.join(DATA_ROOT, "Task_A", "val"),
    )
print("[INFO] Gender model ready.")

# ---------------------- Load FaceNet Model ---------------------
print("[INFO] Loading FaceNet backbone …")
FACENET_W = "facenet_triplet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet = InceptionResnetV1(pretrained=None, classify=False).to(device)
try:
    ckpt = torch.load(FACENET_W, map_location=device)
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith(("last_linear", "last_bn"))}
    facenet.load_state_dict(ckpt, strict=False)
except (FileNotFoundError, RuntimeError):
    DATA_ROOT = r"E:\comsys hackathon\Comys_Hackathon5"
    train_facenet(
        os.path.join(DATA_ROOT, "Task_B", "train"),
        os.path.join(DATA_ROOT, "Task_B", "val"),
    )
    ckpt = torch.load(FACENET_W, map_location=device)
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith(("last_linear", "last_bn"))}
    facenet.load_state_dict(ckpt, strict=False)
facenet.eval()
print("[INFO] FaceNet ready.")

# -------------------- Frame Processing ------------------------

def process_frame(frame: np.ndarray, frame_idx: int, csv_writer=None):
    faces = RetinaFace.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not faces or not isinstance(faces, dict):
        return frame

    for fid, info in faces.items():
        x1, y1, x2, y2 = info["facial_area"]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        face = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_res = cv2.resize(face_rgb, (128, 128))

        # Gender prediction
        pred = gender_model.predict(preprocess_input(np.expand_dims(face_res, 0)), verbose=0)[0]
        idx = int(np.argmax(pred))
        label = "Male" if idx == 0 else "Female"
        prob = pred[idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} ({prob:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            lineType=cv2.LINE_AA
        )

        if csv_writer:
            csv_writer.writerow([frame_idx, fid, label, round(prob, 2)])

        ft = torch.tensor(face_res).permute(2, 0, 1).unsqueeze(0).float()
        ft = (ft - 127.5) / 128.0
        with torch.no_grad():
            _ = facenet(ft.to(device))
    return frame

# ---------------------- Image Mode -----------------------------
if is_image_file(SOURCE):
    img = cv2.imread(SOURCE)
    if img is None:
        raise IOError(f"Cannot open image: {SOURCE}")
    result = process_frame(img.copy(), 1)
    cv2.imwrite("output.jpg", result)
    cv2.imshow("Result", adaptive_resize(result))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[INFO] Saved annotated image → output.jpg")

# ------------------ Video / Webcam Mode ------------------------
else:
    cap = cv2.VideoCapture(0 if SOURCE == "0" else SOURCE)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {SOURCE}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output.avi", fourcc, fps, (w, h))

    csv_f = open("predictions.csv", "w", newline="")
    csv_w = csv.writer(csv_f); csv_w.writerow(["Frame", "Face_ID", "Gender", "Prob"])

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        frame = process_frame(frame, idx, csv_w)
        out.write(frame)
        cv2.imshow("AION", adaptive_resize(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release(); out.release(); csv_f.close(); cv2.destroyAllWindows()
    print("[INFO] Saved annotated video → output.avi & predictions.csv")
