import os
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import joblib
import easyocr
from bnunicodenormalizer import Normalizer
from spellchecker import SpellChecker

from memeClassifier import logger
from memeClassifier.config.configuration import ConfigurationManager


# ==========================================
# 1. Base Model Architectures
# ==========================================

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


# ==========================================
# 2. Prediction Pipeline Implementation
# ==========================================

class MemePredictionPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device} for inference.")
        
        # Initialize Configuration Manager
        self.config_manager = ConfigurationManager()
        
        # Load processors & text normalizers
        logger.info("Initializing OCR Engine and text processors...")
        self.ocr_reader = easyocr.Reader(['bn', 'en'], gpu=torch.cuda.is_available())
        self.bnorm = Normalizer()
        self.spell = SpellChecker()
        
        # Dynamic extraction of model paths and parameters from configs
        try:
            self.political_words_path = self.config_manager.get_find_political_word_config().output_csv
            self.lr_model_path = self.config_manager.get_classification_model_using_text_config().model_path
            self.nn_model_path = self.config_manager.get_neural_network_model_using_text_config().model_path
            self.clip_model_path = self.config_manager.get_clip_model_config().model_save_path
            self.meta_model_path = self.config_manager.get_ensemble_config().meta_model_path
            self.nn_hidden_dim = self.config_manager.params.NeuralNetworkModelUsingText.HIDDEN_DIM
        except Exception as e:
            logger.warning(f"Failed to fetch paths via ConfigurationManager, applying relative fallbacks: {e}")
            self.political_words_path = "artifacts/find_political_word/political_specific_words.csv"
            self.lr_model_path = "artifacts/classification_model_using_text/model.joblib"
            self.nn_model_path = "artifacts/neural_network_model_using_text/model.pth"
            self.clip_model_path = "artifacts/clip_model/model.pth"
            self.meta_model_path = "artifacts/ensemble/meta_model.joblib"
            self.nn_hidden_dim = 64

        # Load models and keywords
        self.political_words = self._load_political_keywords(top_n=500)
        self._load_all_models()

    def _load_political_keywords(self, top_n=500):
        if not os.path.exists(self.political_words_path):
            logger.warning(f"Political words vocabulary missing at {self.political_words_path}. Fallback to empty set.")
            return set()
        pol_words_df = pd.read_csv(self.political_words_path)
        if 'ratio' in pol_words_df.columns:
            pol_words_df = pol_words_df.sort_values('ratio', ascending=False)
        return set(pol_words_df['word'].head(top_n).str.lower().tolist())

    def _load_all_models(self):
        logger.info("Loading baseline trained models into system memory...")
        
        # 1. Load Logistic Regression
        if os.path.exists(self.lr_model_path):
            self.lr_model = joblib.load(self.lr_model_path)
        else:
            logger.error(f"Critical Model Missing: Text Logistic Regression model at {self.lr_model_path}")
            self.lr_model = None

        # 2. Load PyTorch Custom Text Neural Network
        self.nn_model = SimpleNN(input_dim=2, hidden_dim=self.nn_hidden_dim).to(self.device)
        if os.path.exists(self.nn_model_path):
            self.nn_model.load_state_dict(torch.load(self.nn_model_path, map_location=self.device))
            self.nn_model.eval()
        else:
            logger.error(f"Critical Model Missing: PyTorch Text NN model at {self.nn_model_path}")

        # 3. Load Vision-based Fine-tuned CLIP Classifier
        clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPClassifier(clip_base, num_classes=2)
        if os.path.exists(self.clip_model_path):
            self.clip_model.load_state_dict(torch.load(self.clip_model_path, map_location=self.device))
            self.clip_model.to(self.device)
            self.clip_model.eval()
        else:
            logger.error(f"Critical Model Missing: Fine-tuned CLIP model at {self.clip_model_path}")

        # 4. Load Meta Stacking Ensemble Classifier
        if os.path.exists(self.meta_model_path):
            self.meta_model = joblib.load(self.meta_model_path)
            logger.info("✓ Meta Stacking Stacking model loaded successfully.")
        else:
            raise FileNotFoundError(f"Critical Stacking Ensemble Model Missing at {self.meta_model_path}")

    def preprocess_text(self, text):
        if not isinstance(text, str) or text == '':
            return ""
        
        text = text.lower()
        text = re.sub(r'[\|\{\}\[\]\(\);]+', ' ', text)
        text = re.sub(r'\b\w*\d+_\w*\b', ' ', text)
        text = re.sub(r'\b\d+_\d+\b', ' ', text)
        text = re.sub(r'\b\w*\d+\w*_\b', ' ', text)
        text = re.sub(r'\b_\d+\w*\b', ' ', text)
        text = re.sub(r'\b\d+[a-z]*\b', ' ', text)
        text = re.sub(r'\b[a-z]*\d+\b', ' ', text)
        text = re.sub(r'\b[\u09E6-\u09EF]+\b', ' ', text)
        text = re.sub(r'\b[\u09E6-\u09EF]+[\u0980-\u09FF]+\b', ' ', text)
        text = re.sub(r'\b[\u0980-\u09FF]+[\u09E6-\u09EF]+\b', ' ', text)
        text = re.sub(r'\b[\u0980-\u09FF]*[\u09E6-\u09EF]+[\u0980-\u09FF]+[\u09E6-\u09EF]*\b', ' ', text)
        text = re.sub(r'[^\w\s\u0980-\u09FF.,!?]', ' ', text)
        text = re.sub(r'\b[a-z]\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[\u0980-\u09FF]\b', ' ', text)

        words = text.split()
        cleaned_words = []
        
        for word in words:
            if len(word) < 2 or word.strip('.,!?') == '':
                continue
            if '_' in word or re.search(r'\d', word):
                continue
            if re.search(r'[\u09E6-\u09EF]', word):
                continue
            has_bengali = bool(re.search(r'[\u0980-\u09FF]', word))
            if has_bengali:
                try:
                    normalized = self.bnorm(word)['normalized']
                    if normalized and len(normalized) > 1:
                        cleaned_words.append(normalized)
                except:
                    if len(word) > 1:
                        cleaned_words.append(word)
            else:
                word_clean = word.strip('.,!?')
                if len(word_clean) > 2:
                    try:
                        corrected = self.spell.correction(word_clean)
                        if corrected and corrected != word_clean:
                            if corrected in self.spell:
                                cleaned_words.append(corrected)
                            else:
                                cleaned_words.append(word_clean)
                        else:
                            cleaned_words.append(word_clean)
                    except:
                        cleaned_words.append(word_clean)
        
        text = ' '.join(cleaned_words)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

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

    def predict(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at coordinates: {image_path}")

        # --- STEP 1: Text Extraction (OCR) ---
        logger.info("Extracting textual content from target image...")
        result = self.ocr_reader.readtext(image_path)
        detected_texts = []
        for detection in result:
            if isinstance(detection, (tuple, list)) and len(detection) >= 2:
                if isinstance(detection[1], str):
                    detected_texts.append(detection[1])
        raw_text = ' '.join(detected_texts)
        logger.info(f"Raw OCR Output: '{raw_text}'")

        # --- STEP 2: Text Preprocessing ---
        processed_text = self.preprocess_text(raw_text)
        logger.info(f"Cleaned Token Text: '{processed_text}'")

        # --- STEP 3: Feature Counts and Metrics Calculation ---
        words = processed_text.lower().split()
        word_count = len(words)
        matches = sum(1 for word in words if word in self.political_words)
        ratio = matches / word_count if word_count > 0 else 0.0
        
        text_features = np.array([[matches, ratio]], dtype=np.float32)
        logger.info(f"Engineered Count Features -> Matches: {matches}, Density Ratio: {ratio:.4f}")

        # --- STEP 4: Base Probabilities Extraction ---
        # A. Logistic Regression
        if self.lr_model:
            lr_proba = self.lr_model.predict_proba(text_features)
        else:
            lr_proba = np.array([[0.5, 0.5]])

        # B. PyTorch Simple Text NN
        X_tensor = torch.FloatTensor(text_features).to(self.device)
        with torch.no_grad():
            nn_out = self.nn_model(X_tensor).cpu().numpy()
        probs_pol = nn_out
        probs_nonpol = 1.0 - probs_pol
        nn_proba = np.hstack((probs_nonpol, probs_pol))

        # C. Transformers Fine-tuned Vision CLIP
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to read image for CLIP pipeline, processing fallback patch: {e}")
            img = Image.new('RGB', (224, 224), (128, 128, 128))
            
        clip_inputs = self.clip_processor(images=img, return_tensors="pt")
        pixel_values = clip_inputs['pixel_values'].to(self.device)
        with torch.no_grad():
            clip_logits = self.clip_model(pixel_values)
            clip_proba = torch.softmax(clip_logits, dim=1).cpu().numpy()

        # --- STEP 5: Create Meta Stacking Features Matrix ---
        stacking_vector = self.create_stacking_features(lr_proba, nn_proba, clip_proba)

        # --- STEP 6: Execute Meta Stacking Classification Decision ---
        final_prediction = self.meta_model.predict(stacking_vector)[0]
        final_proba = self.meta_model.predict_proba(stacking_vector)[0]
        
        label_mapping = {0: "NonPolitical", 1: "Political"}
        predicted_label = label_mapping[final_prediction]
        confidence = final_proba[final_prediction]

        print("\n" + "="*50)
        print(f"🌟 FINAL ENSEMBLE PREDICTION SUMMARY 🌟")
        print("="*50)
        print(f"Target Meme File : {os.path.basename(image_path)}")
        print(f"Classification   : {predicted_label}")
        print(f"Confidence Score : {confidence * 100:.2f}%")
        print("-"*50)
        print(f"[Base Model Probabilities (Non-Political vs Political)]")
        print(f"  └─ Text Logistic Regression : {lr_proba[0][0]:.4f} vs {lr_proba[0][1]:.4f}")
        print(f"  └─ Text Deep Neural Network : {nn_proba[0][0]:.4f} vs {nn_proba[0][1]:.4f}")
        print(f"  └─ Fine-Tuned Vision CLIP   : {clip_proba[0][0]:.4f} vs {clip_proba[0][1]:.4f}")
        print("="*50 + "\n")

        return predicted_label, confidence


# ==========================================
# 3. CLI Script Runner Initialization
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Bangla Political Meme Classifier Stacking Inference Engine.")
    parser.add_argument("--image", type=str, required=True, help="Complete server absolute filepath pointing to target image.")
    args = parser.parse_args()

    try:
        pipeline = MemePredictionPipeline()
        pipeline.predict(image_path=args.image)
    except Exception as e:
        logger.exception(f"Inference run broken due to system error: {e}")