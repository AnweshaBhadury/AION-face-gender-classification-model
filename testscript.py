import os
import json
import torch
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

class ImageDataset(Dataset):
    """Custom dataset to handle image loading more robustly"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

def load_images_from_folder(folder, class_label=None):
    """Load images from folder with robust error handling"""
    image_paths = []
    labels = []
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")
    
    for file in os.listdir(folder):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(folder, file)
        image_paths.append(img_path)
        if class_label is not None:
            labels.append(class_label)
    
    return image_paths, labels

def evaluate_gender_model(model_path, val_dir):
    """Evaluate the gender classification model with improved image loading"""
    # Load the model
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load validation data
    class_names = ['male', 'female']
    all_image_paths = []
    all_labels = []
    
    for label, class_name in enumerate(class_names):
        folder = os.path.join(val_dir, class_name)
        image_paths, labels = load_images_from_folder(folder, label)
        all_image_paths.extend(image_paths)
        all_labels.extend(labels)

    if not all_image_paths:
        raise ValueError("No valid images found in the dataset directory.")

    # Preprocess images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy().transpose(1, 2, 0))
                          ])  # Convert to HWC format
    
    dataset = ImageDataset(all_image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    images = []
    for batch in dataloader:
        if batch is not None:
            images.append(batch)
    
    if not images:
        raise ValueError("No valid images could be loaded.")
    
    images = np.concatenate(images, axis=0)
    images = preprocess_input(images)
    labels = np.array(all_labels)

    # Make predictions
    predictions = np.argmax(model.predict(images), axis=1)

    # Calculate metrics
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def evaluate_facenet_model(model_path, train_dir, distorted_base_dir, batch_size=32):
    """Evaluate the FaceNet model with architecture matching the saved weights"""
    print("[Task B] Loading FaceNet model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with the correct output dimension (128)
    model = InceptionResnetV1(pretrained='vggface2', classify=False)
    
    # Modify the last layers to match your saved model
    model.last_linear = torch.nn.Linear(1792, 128)
    model.last_bn = torch.nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True)
    
    # Load the state dict with strict=False to ignore size mismatches
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        
        # Remove unexpected keys (like 'last_linear.bias') if present
        if 'last_linear.bias' in state_dict:
            del state_dict['last_linear.bias']
            
        model.load_state_dict(state_dict, strict=False)
    
    model = model.eval().to(device)

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Prepare train embeddings
    print("[Task B] Generating embeddings for reference images...")
    train_embeddings = {}
    
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        image_paths, _ = load_images_from_folder(class_dir)
        dataset = ImageDataset(image_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        class_embeddings = []
        for img in tqdm(dataloader, desc=f"Processing {class_name}"):
            if img is None:
                continue
                
            with torch.no_grad():
                embedding = model(img.to(device)).cpu().numpy()
                class_embeddings.append(embedding)
        
        if class_embeddings:
            train_embeddings[class_name] = np.concatenate(class_embeddings, axis=0)

    # Rest of the function remains the same...
    # [Keep all the code from the distorted images processing onward]

    # Prepare distorted images
    print("[Task B] Loading distorted images...")
    distorted_image_paths = []
    ground_truth = []
    
    for root, _, files in os.walk(distorted_base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                distorted_image_paths.append(os.path.join(root, file))
                ground_truth.append(os.path.basename(root))

    # Process distorted images
    predictions = []
    valid_ground_truth = []
    
    dataset = ImageDataset(distorted_image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("[Task B] Matching distorted images...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch is None:
            continue
            
        current_batch_size = batch.size(0)
        batch_ground_truth = ground_truth[batch_idx*batch_size : batch_idx*batch_size + current_batch_size]
        
        with torch.no_grad():
            batch_embeddings = model(batch.to(device)).cpu().numpy()

        for i, embedding in enumerate(batch_embeddings):
            min_distance = float('inf')
            predicted_label = None
            
            for class_name, ref_embeddings in train_embeddings.items():
                distances = np.linalg.norm(ref_embeddings - embedding, axis=1)
                min_class_distance = np.min(distances)
                
                if min_class_distance < min_distance:
                    min_distance = min_class_distance
                    predicted_label = class_name
            
            if predicted_label is not None:
                predictions.append(predicted_label)
                valid_ground_truth.append(batch_ground_truth[i])

    if not predictions or not valid_ground_truth:
        raise ValueError("No valid predictions or ground truth to evaluate.")

    # Calculate metrics
    top_1_accuracy = accuracy_score(valid_ground_truth, predictions)
    macro_f1_score = f1_score(valid_ground_truth, predictions, average='macro')
    precision = precision_score(valid_ground_truth, predictions, average='macro')
    recall = recall_score(valid_ground_truth, predictions, average='macro')

    return {
        "top_1_accuracy": top_1_accuracy,
        "macro_f1_score": macro_f1_score,
        "precision": precision,
        "recall": recall
    }

if __name__ == "__main__":
    gender_model_path = "gender_classification_efficientnet_final.keras"
    facenet_model_path = "facenet_triplet.pth"
    gender_val_dir = r"E:\comsys hackathon\Comys_Hackathon5\Task_A\val"
    facenet_train_dir = r"E:\comsys hackathon\Comys_Hackathon5\Task_B\train"
    facenet_distorted_dir = r"E:\comsys hackathon\Comys_Hackathon5\Task_B"

    # Initialize metrics
    gender_metrics = None
    facenet_metrics = None

    # Evaluate Task A
    try:
        gender_metrics = evaluate_gender_model(gender_model_path, gender_val_dir)
        print("Gender Classification Metrics:", gender_metrics)
    except Exception as e:
        print(f"Error in Gender Classification Evaluation: {e}")

    # Evaluate Task B
    try:
        facenet_metrics = evaluate_facenet_model(facenet_model_path, facenet_train_dir, facenet_distorted_dir)
        print("Face Matching Metrics:", facenet_metrics)
    except Exception as e:
        print(f"Error in Face Matching Evaluation: {e}")

    # Save results
    results = {}
    if gender_metrics:
        results["gender_classification"] = gender_metrics
    if facenet_metrics:
        results["face_matching"] = facenet_metrics

    if results:
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("Evaluation results saved to 'evaluation_results.json'.")