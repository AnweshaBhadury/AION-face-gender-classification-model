import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt

# Triplet Loss implementation (unchanged)
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

# Train FaceNet function (with fixed scheduler)
def train_facenet(data_dir, val_dir, epochs=10, batch_size=32, embedding_dim=128):
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize FaceNet model
    model = InceptionResnetV1(pretrained='vggface2', classify=False)
    model.last_linear = nn.Linear(model.last_linear.in_features, embedding_dim)
    model.last_bn = nn.BatchNorm1d(embedding_dim)
    model = model.to(device)
    model.train()

    # Loss function and optimizer
    triplet_loss = TripletLoss(margin=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Fixed scheduler without verbose parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Metrics storage
    train_losses = []
    val_losses = []

    # Training loop
    print("Starting FaceNet training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, _ in train_loader:
            if inputs.size(0) < 3 or inputs.size(0) % 3 != 0:
                continue

            inputs = inputs.to(device)
            split_size = inputs.size(0) // 3
            anchor = inputs[:split_size]
            positive = inputs[split_size:2 * split_size]
            negative = inputs[2 * split_size:3 * split_size]

            optimizer.zero_grad()
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                if inputs.size(0) < 3:
                    continue
                    
                inputs = inputs.to(device)
                split_size = inputs.size(0) // 3
                anchor = inputs[:split_size]
                positive = inputs[split_size:2 * split_size]
                negative = inputs[2 * split_size:3 * split_size]
                
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                val_loss += triplet_loss(anchor_emb, positive_emb, negative_emb).item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save the trained model
    model_path = 'facenet_triplet.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}.")

    # Save training history
    history_path = 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f)
    print(f"Training history saved to {history_path}.")

    # Plot metrics
    plot_metrics(train_losses, val_losses)

# Plot metrics function (updated for validation)
def plot_metrics(train_losses, val_losses=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

# Main function (unchanged)
if __name__ == "__main__":
    train_dir = r"E:\comsys hackathon\Comys_Hackathon5\Task_B\train"
    val_dir = r"E:\comsys hackathon\Comys_Hackathon5\Task_B\val"
    
    # Define parameters
    epochs = 10  # Number of training epochs
    batch_size = 32  # Batch size for training

    # Call the training function
    train_facenet(train_dir, val_dir, epochs, batch_size)