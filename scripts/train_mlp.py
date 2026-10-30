import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class StutterDataset(Dataset):
    def __init__(self, df, embeddings_dir, desc="Loading"):
        self.df = df.reset_index(drop=True)
        self.embeddings_dir = embeddings_dir
        
        self.data = []
        self.labels = []
        
        for idx in tqdm(range(len(self.df)), desc=f"To RAM [{desc}]", leave=False):
            file_name = self.df.loc[idx, 'file_name']
            npy_name = file_name.replace(".wav", ".npy")
            embedding_path = self.embeddings_dir / npy_name
            
            embedding = np.load(str(embedding_path))
            label = self.df.loc[idx, 'is_stutter']
            
            # Memory optimization: store as float16 in RAM
            self.data.append(embedding.astype(np.float16))
            self.labels.append(label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class AudioClassifier(nn.Module):
    def __init__(self, input_dim=1024):
        super(AudioClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

def train_model():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    embeddings_dir = base_dir / "data" / "sep28k_embeddings"
    csv_path = base_dir / "data" / "sep28k_audio" / "metadata.csv"
    
    print("\n" + "="*60)
    print("[INFO] Initializing data and loading metadata...")
    print("="*60)
    
    df = pd.read_csv(csv_path)

    tqdm.pandas(desc="Checking .npy files", leave=False)
    df['npy_exists'] = df['file_name'].progress_apply(lambda x: (embeddings_dir / x.replace('.wav', '.npy')).exists())
    df = df[df['npy_exists'] == True]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_stutter'])
    
    train_dataset = StutterDataset(train_df, embeddings_dir, desc="Train")
    val_dataset = StutterDataset(val_df, embeddings_dir, desc="Valid")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Compute device: {device.type.upper()}")
    
    model = AudioClassifier().to(device)
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    epochs = 40
    best_roc_auc = 0.0
    
    print("\n" + "="*60)
    print("[START] Starting MLP classifier training loop...")
    print("="*60 + "\n")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", leave=False)
        
        for embeddings, labels in train_loop:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
            
        model.eval()
        val_preds = []
        val_probs = []
        val_true = []
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Valid]", leave=False)
        
        with torch.no_grad():
            for embeddings, labels in val_loop:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        val_prec = precision_score(val_true, val_preds, zero_division=0)
        val_rec = recall_score(val_true, val_preds, zero_division=0)
        val_roc = roc_auc_score(val_true, val_probs)
        
        avg_train_loss = train_loss / len(train_loader)
        
        scheduler.step(val_roc)
        
        epoch_duration = time.time() - epoch_start_time
        mins, secs = divmod(epoch_duration, 60)
        
        print(f"Epoch {epoch+1:02d}/{epochs} [{int(mins):02d}:{int(secs):02d}] | Loss: {avg_train_loss:.4f} | Val ROC-AUC: {val_roc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
        
        if val_roc > best_roc_auc:
            best_roc_auc = val_roc
            torch.save(model.state_dict(), base_dir / "scripts" / "best_mlp_model.pth")
            print(f"  [SAVE] New best model saved! (ROC-AUC: {best_roc_auc:.4f})")

    print("\n" + "="*60)
    print(f"[DONE] Training complete. Best validation ROC-AUC: {best_roc_auc:.4f}")
    print("="*60)

if __name__ == "__main__":
    train_model()