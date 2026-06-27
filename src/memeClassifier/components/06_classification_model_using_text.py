import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from memeClassifier import logger
from memeClassifier.entity.config_entity import ClassificationModelUsingTextConfig

class ClassificationModelUsingText:
    def __init__(self, config: ClassificationModelUsingTextConfig):
        self.config = config

    def initiate_model_training(self):
        logger.info("Loading feature datasets")
        train_df = pd.read_csv(self.config.train_features_csv)
        
        X = train_df[['political_specific_count', 'political_specific_ratio']]
        y = train_df['Label'].map({'Political': 1, 'NonPolitical': 0})
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Validation set size: {X_val.shape[0]}")
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Classification Report (Validation Set):\n{classification_report(y_val, y_val_pred, target_names=['NonPolitical', 'Political'])}")
        
        joblib.dump(model, self.config.model_path)
        logger.info(f"Model saved to {self.config.model_path}")