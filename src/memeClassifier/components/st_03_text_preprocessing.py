import re
import pandas as pd
from bnunicodenormalizer import Normalizer
from spellchecker import SpellChecker
from memeClassifier import logger
from memeClassifier.entity.config_entity import TextPreprocessingConfig

class TextPreprocessing:
    def __init__(self, config: TextPreprocessingConfig):
        self.config = config
        self.bnorm = Normalizer()
        self.spell = SpellChecker()

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

    def initiate_text_preprocessing(self):
        logger.info("Starting text preprocessing for train and test datasets")
        
        train_df = pd.read_csv(self.config.input_train_csv)
        test_df = pd.read_csv(self.config.input_test_csv)
        
        logger.info("Preprocessing train dataset...")
        train_df['Processed_Text'] = train_df['Extracted_Text'].apply(self.preprocess_text)
        
        logger.info("Preprocessing test dataset...")
        test_df['Processed_Text'] = test_df['Extracted_Text'].apply(self.preprocess_text)
        
        train_df.to_csv(self.config.output_train_csv, index=False)
        test_df.to_csv(self.config.output_test_csv, index=False)
        
        logger.info(f"Saved processed train dataset to {self.config.output_train_csv}")
        logger.info(f"Saved processed test dataset to {self.config.output_test_csv}")