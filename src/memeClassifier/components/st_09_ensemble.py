import os
from typing import cast
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
from memeClassifier import logger
from memeClassifier.entity.config_entity import EnsembleConfig, SolverType


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2, dropout=0.3):
        super(CLIPClassifier, self).__init__()
        self.clip = clip_model
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

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class EnsembleModel:
    def __init__(self, config: EnsembleConfig, nn_hidden_dim: int = 64):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_hidden_dim = nn_hidden_dim

    def get_nn_predictions(self, X_features, model, batch_size=16):
        model.eval()
        all_probs = []
        X_tensor = torch.FloatTensor(X_features).to(self.device)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X_features), batch_size), desc="NN predictions"):
                batch_X = X_tensor[i:i + batch_size]
                outputs = model(batch_X)
                probs_pol = outputs.cpu().numpy()
                probs_nonpol = 1.0 - probs_pol
                batch_probs = np.hstack((probs_nonpol, probs_pol))
                all_probs.append(batch_probs)
        return np.vstack(all_probs)

    def get_image_predictions(self, image_names, img_dir, model, processor, batch_size=16):
        model.eval()
        all_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_names), batch_size), desc="Image predictions"):
                batch_names = image_names[i:i + batch_size]
                batch_images = []
                for img_name in batch_names:
                    img_path = os.path.join(str(img_dir), img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                    except:
                        img = Image.new('RGB', (224, 224), (128, 128, 128))
                    batch_images.append(img)
                inputs = processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                outputs = model(pixel_values)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)

    def create_stacking_features(self, lr_proba, nn_proba, clip_proba):
        # Base probabilities
        features = [
            lr_proba[:, 0], lr_proba[:, 1],
            nn_proba[:, 0], nn_proba[:, 1],
            clip_proba[:, 0], clip_proba[:, 1],
        ]
        
        # Confidence
        features.extend([
            np.max(lr_proba, axis=1),
            np.max(nn_proba, axis=1),
            np.max(clip_proba, axis=1)
        ])
        
        # Agreement
        lr_pred = np.argmax(lr_proba, axis=1)
        nn_pred = np.argmax(nn_proba, axis=1)
        clip_pred = np.argmax(clip_proba, axis=1)
        
        features.extend([
            (lr_pred == nn_pred).astype(float),
            (lr_pred == clip_pred).astype(float),
            (nn_pred == clip_pred).astype(float)
        ])
        
        # Differences
        features.extend([
            np.abs(lr_proba[:, 1] - nn_proba[:, 1]),
            np.abs(lr_proba[:, 1] - clip_proba[:, 1]),
            np.abs(nn_proba[:, 1] - clip_proba[:, 1])
        ])
        
        # Statistical
        all_pol_probs = np.column_stack([lr_proba[:, 1], nn_proba[:, 1], clip_proba[:, 1]])
        features.extend([
            np.mean(all_pol_probs, axis=1),
            np.max(all_pol_probs, axis=1),
            np.min(all_pol_probs, axis=1)
        ])
        
        return np.column_stack(features)

    def initiate_ensemble_training(self):
        train_features_df = pd.read_csv(self.config.train_features_csv)
        
        # The train features CSV contains the Label mapping directly
        train_split, val_split = train_test_split(
            train_features_df, test_size=0.2, random_state=42, stratify=train_features_df['Label']
        )
        
        y_val = val_split['Label'].map({'Political': 1, 'NonPolitical': 0}).values
        X_val_features = val_split[['political_specific_count', 'political_specific_ratio']].values
        
        # We also need to get the image names corresponding to these validation samples
        # Fortunately, the features CSV should align with train_dataset_cleaned.csv row by row
        train_raw_df = pd.read_csv(self.config.train_csv_path)
        val_images = train_raw_df.iloc[val_split.index]['Image_name'].tolist()

        logger.info("Loading base models...")
        
        # Load LR
        lr_model = None
        if os.path.exists(self.config.logistic_regression_model_path):
            lr_model = joblib.load(self.config.logistic_regression_model_path)
            logger.info("Loaded Logistic Regression model.")
        else:
            logger.warning(f"LR model missing at {self.config.logistic_regression_model_path}")
            
        # Load NN
        nn_model = SimpleNN(input_dim=2, hidden_dim=self.nn_hidden_dim).to(self.device)
        if os.path.exists(self.config.neural_network_model_path):
            nn_model.load_state_dict(torch.load(self.config.neural_network_model_path, map_location=self.device))
            logger.info("Loaded PyTorch Neural Network model.")
        else:
            logger.warning(f"NN model missing at {self.config.neural_network_model_path}")
            
        # Load CLIP
        clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPClassifier(clip_base, num_classes=2)
        if os.path.exists(self.config.clip_model_path):
            clip_model.load_state_dict(torch.load(self.config.clip_model_path, map_location=self.device))
            logger.info("Loaded CLIP model.")
        else:
            logger.warning(f"CLIP model missing at {self.config.clip_model_path}")
        clip_model = clip_model.to(self.device)
        
        logger.info("Generating predictions from base models...")
        
        # Generate LR probabilities
        if lr_model:
            lr_proba_val = lr_model.predict_proba(X_val_features)
        else:
            lr_proba_val = np.ones((len(val_split), 2)) * 0.5
            
        # Generate NN probabilities
        nn_proba_val = self.get_nn_predictions(X_val_features, nn_model)
        
        # Generate CLIP probabilities
        clip_proba_val = self.get_image_predictions(
            val_images, self.config.train_img_dir, clip_model, clip_processor
        )

        logger.info("Creating stacking features...")
        X_val_stacking = self.create_stacking_features(lr_proba_val, nn_proba_val, clip_proba_val)

        logger.info("Training meta-model manually...")
        solver_name = cast(SolverType, self.config.solver)
        meta_model = LogisticRegression(
            C=self.config.c,
            class_weight=self.config.class_weight,
            solver=solver_name,
            max_iter=self.config.max_iter,
            random_state=42
        )
        meta_model.fit(X_val_stacking, y_val)

        ensemble_preds_val = meta_model.predict(X_val_stacking)
        logger.info(f"Ensemble Val Accuracy: {accuracy_score(y_val, ensemble_preds_val):.4f}")
        logger.info(f"Ensemble Val F1: {f1_score(y_val, ensemble_preds_val, average='macro'):.4f}")

        joblib.dump(meta_model, self.config.meta_model_path)
        logger.info(f"Meta-model saved to {self.config.meta_model_path}")