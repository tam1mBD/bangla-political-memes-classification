import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn as mlflow_sklearn
from urllib.parse import urlparse
from memeClassifier import logger
from memeClassifier.entity.config_entity import ModelEvaluationConfig
from memeClassifier.utils.common import save_json

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

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, nn_hidden_dim: int = 64):
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
        features = [
            lr_proba[:, 0], lr_proba[:, 1],
            nn_proba[:, 0], nn_proba[:, 1],
            clip_proba[:, 0], clip_proba[:, 1],
        ]
        features.extend([
            np.max(lr_proba, axis=1),
            np.max(nn_proba, axis=1),
            np.max(clip_proba, axis=1)
        ])
        lr_pred = np.argmax(lr_proba, axis=1)
        nn_pred = np.argmax(nn_proba, axis=1)
        clip_pred = np.argmax(clip_proba, axis=1)
        features.extend([
            (lr_pred == nn_pred).astype(float),
            (lr_pred == clip_pred).astype(float),
            (nn_pred == clip_pred).astype(float)
        ])
        features.extend([
            np.abs(lr_proba[:, 1] - nn_proba[:, 1]),
            np.abs(lr_proba[:, 1] - clip_proba[:, 1]),
            np.abs(nn_proba[:, 1] - clip_proba[:, 1])
        ])
        all_pol_probs = np.column_stack([lr_proba[:, 1], nn_proba[:, 1], clip_proba[:, 1]])
        features.extend([
            np.mean(all_pol_probs, axis=1),
            np.max(all_pol_probs, axis=1),
            np.min(all_pol_probs, axis=1)
        ])
        return np.column_stack(features)

    def evaluate(self):
        logger.info("Loading test datasets")
        test_features_df = pd.read_csv(self.config.test_features_csv, skipinitialspace=True)
        test_features_df.columns = test_features_df.columns.str.strip()
        y_test = test_features_df['Label'].map({'Political': 1, 'NonPolitical': 0}).to_numpy(dtype=np.int64)
        X_test_features = test_features_df[['political_specific_count', 'political_specific_ratio']].to_numpy(dtype=np.float32)
        
        test_raw_df = pd.read_csv(self.config.test_csv_path, skipinitialspace=True)
        test_images = test_raw_df['Image_name'].tolist()

        logger.info("Loading base models for evaluation...")
        
        if os.path.exists(self.config.logistic_regression_model_path):
            lr_model = joblib.load(self.config.logistic_regression_model_path)
        else:
            logger.warning(f"LR model missing at {self.config.logistic_regression_model_path}")
            lr_model = None
            
        nn_model = SimpleNN(input_dim=2, hidden_dim=self.nn_hidden_dim).to(self.device)
        if os.path.exists(self.config.neural_network_model_path):
            nn_model.load_state_dict(torch.load(self.config.neural_network_model_path, map_location=self.device))
        else:
            logger.warning(f"NN model missing at {self.config.neural_network_model_path}")
            
        clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPClassifier(clip_base, num_classes=2)
        if os.path.exists(self.config.clip_model_path):
            clip_model.load_state_dict(torch.load(self.config.clip_model_path, map_location=self.device))
        else:
            logger.warning(f"CLIP model missing at {self.config.clip_model_path}")
        clip_model = clip_model.to(self.device)

        if os.path.exists(self.config.meta_model_path):
            self.meta_model = joblib.load(self.config.meta_model_path)
        else:
            raise FileNotFoundError(f"Meta model missing at {self.config.meta_model_path}")

        logger.info("Generating predictions from base models on test set...")
        if lr_model:
            lr_proba_test = lr_model.predict_proba(X_test_features)
        else:
            lr_proba_test = np.ones((len(y_test), 2)) * 0.5
            
        nn_proba_test = self.get_nn_predictions(X_test_features, nn_model)
        clip_proba_test = self.get_image_predictions(test_images, self.config.test_img_dir, clip_model, clip_processor)

        X_test_stacking = self.create_stacking_features(lr_proba_test, nn_proba_test, clip_proba_test)

        logger.info("Predicting with meta-model...")
        ensemble_preds_test = self.meta_model.predict(X_test_stacking)
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, ensemble_preds_test),
            "f1": f1_score(y_test, ensemble_preds_test, average='macro'),
            "precision": precision_score(y_test, ensemble_preds_test, average='macro'),
            "recall": recall_score(y_test, ensemble_preds_test, average='macro')
        }
        
        logger.info(f"Test Metrics: {self.metrics}")
        save_json(path=Path("scores.json"), data=self.metrics)
        
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # Flatten all params for logging
            flat_params = {}
            for key, val in self.config.all_params.items():
                if isinstance(val, dict) or hasattr(val, '__dict__'):
                    items = val.items() if isinstance(val, dict) else vars(val).items()
                    for sub_k, sub_v in items:
                        flat_params[f"{key}_{sub_k}"] = str(sub_v)
                else:
                    flat_params[key] = str(val)
                    
            mlflow.log_params(flat_params)
            mlflow.log_metrics(self.metrics)
            
            if tracking_url_type_store != "file":
                mlflow_sklearn.log_model(self.meta_model, "model", registered_model_name="MemeEnsembleModel")
            else:
                mlflow_sklearn.log_model(self.meta_model, "model")
            logger.info("Logged to MLflow successfully.")