import pandas as pd
from memeClassifier import logger
from memeClassifier.entity.config_entity import TextAnalysisConfig

class TextAnalysis:
    def __init__(self, config: TextAnalysisConfig):
        self.config = config

    def load_political_specific_words(self, top_n=500):
        pol_words = pd.read_csv(self.config.political_words_csv)
        if 'ratio' in pol_words.columns:
            pol_words = pol_words.sort_values('ratio', ascending=False)
        all_pol_words = set(pol_words['word'].head(top_n).str.lower().tolist())
        return all_pol_words

    def extract_text_features(self, text, political_specific_words):
        if pd.isna(text) or text == '':
            return {'political_specific_count': 0, 'political_specific_ratio': 0.0}
        
        words = str(text).lower().split()
        word_count = len(words)
        political_specific_matches = sum(1 for word in words if word in political_specific_words)
        political_specific_ratio = political_specific_matches / word_count if word_count > 0 else 0.0
        
        return {
            'political_specific_count': political_specific_matches,
            'political_specific_ratio': political_specific_ratio
        }

    def initiate_text_analysis(self):
        logger.info("Starting text analysis")
        
        train_df = pd.read_csv(self.config.input_train_csv)
        test_df = pd.read_csv(self.config.input_test_csv)
        
        political_specific_words = self.load_political_specific_words()
        logger.info(f"Political-specific words loaded: {len(political_specific_words)}")
        
        logger.info("Extracting features from training data...")
        train_features = train_df['Processed_Text'].apply(
            lambda x: self.extract_text_features(x, political_specific_words)
        )
        train_features_df = pd.DataFrame(train_features.tolist())
        train_with_features = pd.concat([train_df, train_features_df], axis=1)
        train_with_features.to_csv(self.config.output_train_features, index=False)
        logger.info(f"Train features saved to {self.config.output_train_features}")
        
        logger.info("Extracting features from testing data...")
        test_features = test_df['Processed_Text'].apply(
            lambda x: self.extract_text_features(x, political_specific_words)
        )
        test_features_df = pd.DataFrame(test_features.tolist())
        test_with_features = pd.concat([test_df, test_features_df], axis=1)
        test_with_features.to_csv(self.config.output_test_features, index=False)
        logger.info(f"Test features saved to {self.config.output_test_features}")