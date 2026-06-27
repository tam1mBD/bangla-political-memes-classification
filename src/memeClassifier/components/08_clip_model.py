import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from memeClassifier import logger
from memeClassifier.entity.config_entity import ClipModelConfig

def preprocess_meme_image_minimal(img_path):
    try:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        return Image.new('RGB', (224, 224), (255, 255, 255))


def validate_dataset(df, img_dir):
    logger.info("Validating dataset...")
    corrupted = []
    missing = []
    for idx, row in df.iterrows():
        img_path = os.path.join(img_dir, row["Image_name"])
        if not os.path.exists(img_path):
            missing.append(row["Image_name"])
            continue
        try:
            img = Image.open(img_path)
            img.verify()
        except:
            corrupted.append(row["Image_name"])

    problematic = set(missing + corrupted)
    if problematic:
        df_clean = df[~df["Image_name"].isin(problematic)].reset_index(drop=True)
        logger.info(f"Cleaned dataset: {len(df_clean)} images (removed {len(problematic)})")
        return df_clean
    return df


class CLIPMemeDataset(Dataset):
    def __init__(self, df, img_dir, processor, train=True, label2id=None):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor
        self.train = train
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image_name"])
        img = preprocess_meme_image_minimal(img_path)
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        if self.train:
            if self.label2id is None:
                raise ValueError("label2id must be provided when training mode is enabled")

            label = self.label2id.get(row["Label"])
            if label is None:
                raise KeyError(f"Label '{row['Label']}' is not present in label mapping")
            return pixel_values, label
        return pixel_values, row["Image_name"]


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2, dropout=0.3):
        super(CLIPClassifier, self).__init__()
        self.clip = clip_model
        for param in self.clip.vision_model.parameters():
            param.requires_grad = False
        hidden_size = self.clip.config.projection_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        image_embeds = self.clip.visual_projection(image_embeds)
        logits = self.classifier(image_embeds)
        return logits

    def unfreeze_vision_model(self):
        for param in self.clip.vision_model.parameters():
            param.requires_grad = True
        logger.info("CLIP vision model unfrozen for fine-tuning")


class ClipModelTraining:
    def __init__(self, config: ClipModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.epochs = 10
        self.lr = 1e-5

    def initiate_clip_model_training(self):
        train_df = pd.read_csv(self.config.train_csv_path)
        train_df = validate_dataset(train_df, str(self.config.train_img_dir))

        classes = sorted(train_df["Label"].unique())
        label2id = {c: i for i, c in enumerate(classes)}

        train_split, val_split = train_test_split(
            train_df, test_size=0.15, random_state=42, stratify=train_df["Label"]
        )

        model_name = "openai/clip-vit-base-patch32"
        clip_model_base = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        train_ds = CLIPMemeDataset(train_split, str(self.config.train_img_dir), processor, True, label2id)
        train_dataset_size = len(train_ds)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

        val_ds = CLIPMemeDataset(val_split, str(self.config.train_img_dir), processor, True, label2id)
        val_dataset_size = len(val_ds)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)

        model = CLIPClassifier(clip_model_base, num_classes=len(classes)).to(self.device)

        class_counts = Counter(train_split["Label"].map(label2id))
        class_weights = torch.tensor(
            [1.0 / class_counts[i] for i in range(len(classes))],
            dtype=torch.float32
        ).to(self.device)
        class_weights = class_weights / class_weights.sum() * len(classes)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        best_val_f1 = 0.0
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            if epoch == 2:
                model.unfreeze_vision_model()

            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            for pixel_values, labels in tqdm(train_loader, desc="Training"):
                pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    outputs = model(pixel_values)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * pixel_values.size(0)
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(train_labels, train_preds)
            train_loss /= train_dataset_size

            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for pixel_values, labels in tqdm(val_loader, desc="Validation"):
                    pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)
                    with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                        outputs = model(pixel_values)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item() * pixel_values.size(0)
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            val_loss /= val_dataset_size

            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), self.config.model_save_path)
                logger.info(f"  Best model saved! (Val F1: {best_val_f1:.4f})")

        logger.info("Training complete.")