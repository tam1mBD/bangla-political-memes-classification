import easyocr
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
from memeClassifier import logger
from memeClassifier.entity.config_entity import TextExtractionConfig

class TextExtraction:
    def __init__(self, config: TextExtractionConfig):
        self.config = config
        logger.info("Initializing EasyOCR reader for Bengali and English...")
        self.reader = easyocr.Reader(['bn', 'en'], gpu=False)
        logger.info("✓ Reader initialized")
        
    def extract_text_and_save(self, csv_path: Path, image_folder: Path, output_path: Path):
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} images from {csv_path}")
        
        extracted_texts = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            image_name = row['Image_name']
            image_path = os.path.join(image_folder, image_name)
            
            try:
                if os.path.exists(image_path):
                    result = self.reader.readtext(image_path)
                    detected_texts = []
                    for detection in result:
                        if isinstance(detection, (tuple, list)) and len(detection) >= 2:
                            text_value = detection[1]
                            if isinstance(text_value, str):
                                detected_texts.append(text_value)
                        elif isinstance(detection, dict):
                            text_value = detection.get('text')
                            if isinstance(text_value, str):
                                detected_texts.append(text_value)

                    text = ' '.join(detected_texts)
                    extracted_texts.append(text)
                else:
                    logger.warning(f"Image not found: {image_path}")
                    extracted_texts.append("")
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                extracted_texts.append("")
                
        df['Extracted_Text'] = extracted_texts
        df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Saved results to {output_path}")
        logger.info(f"Statistics for {output_path}:")
        logger.info(f"  Total images processed: {len(df)}")
        logger.info(f"  Images with extracted text: {(df['Extracted_Text'] != '').sum()}")
        logger.info(f"  Images with no text: {(df['Extracted_Text'] == '').sum()}")
        
    def initiate_text_extraction(self):
        logger.info("Extracting text for training data")
        self.extract_text_and_save(
            self.config.train_csv_path,
            self.config.train_image_folder,
            self.config.extracted_train_csv
        )
        
        logger.info("Extracting text for testing data")
        self.extract_text_and_save(
            self.config.test_csv_path,
            self.config.test_image_folder,
            self.config.extracted_test_csv
        )