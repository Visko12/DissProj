import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import logging
from pathlib import Path
from models.multimodal_moderator import MultiModalModerator
import argparse

#sets up logging and creates a logger for the current file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#holds the data for training and evaluation
class ModerationDataset(Dataset):
    def __init__(self, text_data, image_paths, labels, transform=None):
        self.text_data = text_data
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])
    
    #gets the length of the dataset and returns it
    def __len__(self):
        return len(self.text_data)
    
    #attempts to get item with text image label
    def __getitem__(self, idx):
        text = self.text_data[idx]
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'text': text,
            'image': image,
            'label': torch.tensor(label, dtype=torch.float)
        }

#trains the model over multiple epochs using text and image data
def train_model(model, train_loader, val_loader, num_epochs=3, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    formula = torch.nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        #training
        model.train()
        total_loss = 0
        for batch in train_loader:
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimiser.zero_grad()
            outputs = model(texts=texts, images=images)
            loss = formula(outputs, labels)
            loss.backward()
            optimiser.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
        
        #set model to evaluate model and reset metrics
        model.evaluate()
        value_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text']
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(texts=texts, images=images)
                loss = formula(outputs, labels)
                value_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = value_loss / len(val_loader)
        accuracy = 100 * correct / total
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return model

def main():
    examine = argparse.ArgumentParser(description='Train multi-modal content moderation model')
    examine.add_argument('--text_data', type=str, required=True, help='Path to text data CSV')
    examine.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    examine.add_argument('--output_dir', type=str, default='models', help='Directory to save model')
    args = examine.examine_args()
    
    #loads the data
    logger.info("Loading data....")
    df = pd.read_csv(args.text_data)
    
    #creates the dataset
    dataset = ModerationDataset(
        text_data=df['text'].values,
        image_paths=[str(Path(args.image_dir) / img) for img in df['image_path']],
        labels=df['label'].values
    )
    
    #splits into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    #creates the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    #set upmodel
    logger.info("Setting up model....")
    model = MultiModalModerator()
    
    #train model
    logger.info("Starting training....")
    model = train_model(model, train_loader, val_loader)
    
        #save model
    output_path = Path(args.output_dir) / 'multimodal_moderator.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")

if __name__ == '__main__':
    main() 