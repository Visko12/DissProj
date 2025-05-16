import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import logging
from models.multimodal_moderator import ContentModerator
import requests
from tqdm import tqdm
import pandas as pd
import zipfile
import shutil

#set up login
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def download_and_prepare_dataset():
    #create directory for dataset
    data_dir = 'data/nsfw_images'
    os.makedirs(data_dir, exist_ok=True)
    
    #download dataset
    logger.info("Downloading NSFW dataset...")
    zip_path = os.path.join(data_dir, 'nsfw_dataset.zip')
    if not os.path.exists(zip_path):
        #download link straight from github
        url = 'https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_model.onnx'
        download_file(url, zip_path)
    
    #extract dataset
    logger.info("Extracting dataset...")
    extract_dir = os.path.join(data_dir, 'extracted')
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    #process images and create metadata
    logger.info("Processing images and creating metadata...")
    metadata = []
    
    #process each category
    for category in ['sfw', 'nsfw']:
        category_dir = os.path.join(extract_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        #set label to 1 for nsfw and 0 for sfw
        label = 1 if category == 'nsfw' else 0
        
        #process the images
        for idx, filename in enumerate(tqdm(os.listdir(category_dir), desc=f"Processing {category}")):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            #assign the data to train and create the folder
            split = 'train' if idx % 10 < 8 else 'validation' if idx % 10 < 9 else 'test'
            split_dir = os.path.join(data_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            #copy and rename image
            src_path = os.path.join(category_dir, filename)
            dst_path = os.path.join(split_dir, f'{category}_{idx}.jpg')
            shutil.copy2(src_path, dst_path)
            
            #add to metadata
            metadata.append({
                'image_path': dst_path,
                'label': label,
                'category': category,
                'split': split
            })
    
    #save metadata
    df = pd.DataFrame(metadata)
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")
    
    #clean up extracted and zipped files if they exist and then return the data directory
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    return data_dir

class NSFWDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
        
        #load dataset from Csv file
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        #filter metadata for the split and extract images
        self.metadata = self.metadata[self.metadata['split'] == split]
        
        self.image_paths = self.metadata['image_path'].values
        self.labels = self.metadata['label'].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return torch.zeros(3, 224, 224), label

def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    formula = torch.nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        #train the model for one epoch and calculate the loss and accuracy
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimiser.zero_grad()
            outputs = model(images)
            loss = formula(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimiser.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted.squeeze() == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        #validate the model for one epoch and calculate the loss and accuracy
        model.evaluate()
        value_loss = 0
        value_correct = 0
        value_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = formula(outputs, labels.float().unsqueeze(1))
                
                value_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                value_correct += (predicted.squeeze() == labels).sum().item()
                value_total += labels.size(0)
        
        value_loss = value_loss / len(val_loader)
        val_acc = value_correct / value_total
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'Val Loss: {value_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        #save best model
        if value_loss < best_val_loss:
            best_val_loss = value_loss
            os.makedirs('models/saved', exist_ok=True)
            torch.save(model.state_dict(), 'models/saved/best_nsfw_model.pth')
            logger.info('Saved best model')

def main():
    #download and prepare dataset
    data_dir = download_and_prepare_dataset()
    
    #create datasets
    logger.info("Creating datasets...")
    train_dataset = NSFWDataset(data_dir, split='train')
    val_dataset = NSFWDataset(data_dir, split='validation')
    
    #create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    
    #set up the model for content moderation
    model = ContentModerator()
    
    #train the model
    logger.info("Starting training....")
    train_model(model, train_loader, val_loader)
    logger.info("Training complete!")

if __name__ == '__main__':
    main() 