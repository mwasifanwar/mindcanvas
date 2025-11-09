# data/art_loader.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class ArtTherapyDataset(Dataset):
    def __init__(self, image_dir, annotations_file=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        if annotations_file and os.path.exists(annotations_file):
            self.annotations = pd.read_csv(annotations_file)
        else:
            self.annotations = self._generate_synthetic_annotations()
            
    def _generate_synthetic_annotations(self):
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        annotations = []
        for img_file in image_files:
            annotation = {
                'image_path': img_file,
                'emotion': np.random.choice(['joy', 'sadness', 'anger', 'calm']),
                'intensity': np.random.uniform(0.1, 1.0),
                'color_dominance': np.random.choice(['warm', 'cool', 'neutral']),
                'style': np.random.choice(['abstract', 'expressionist', 'realistic']),
                'therapeutic_value': np.random.uniform(0.3, 0.9)
            }
            annotations.append(annotation)
            
        return pd.DataFrame(annotations)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_info = self.annotations.iloc[idx]
        img_path = os.path.join(self.image_dir, img_info['image_path'])
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        emotion_label = self._encode_emotion(img_info['emotion'])
        intensity = torch.tensor(img_info['intensity'], dtype=torch.float32)
        
        return {
            'image': image,
            'emotion': emotion_label,
            'intensity': intensity,
            'metadata': img_info
        }
    
    def _encode_emotion(self, emotion):
        emotion_map = {emotion: idx for idx, emotion in enumerate(config.EMOTION_LABELS)}
        return torch.tensor(emotion_map.get(emotion, 0), dtype=torch.long)

class ArtDataLoader:
    def __init__(self, image_size=224, batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        
    def get_transforms(self, train=True):
        from torchvision import transforms
        
        if train:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def create_data_loaders(self, data_dir, annotations_file=None, train_ratio=0.8):
        full_dataset = ArtTherapyDataset(data_dir, annotations_file, self.get_transforms(train=True))
        
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = self.get_transforms(train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def load_single_image(self, image_path):
        transform = self.get_transforms(train=False)
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)