import pandas as pd
import numpy as np
from collections import Counter

from memeClassifier.entity.config_entity import FindPoliticalWordConfig

class FindPoliticalWord:
    def __init__(self, config: FindPoliticalWordConfig):
        self.config = config

    def extract_words(self, text):
        # Convert to lowercase and split by whitespace
        words = str(text).lower().split()
        # Remove words that are too short (less than 3 characters)
        words = [w for w in words if len(w) >= 3]
        return words

    def process(self):
        df = pd.read_csv(self.config.input_train_csv)
        print(f"Dataset shape: {df.shape}")
        
        political_text = df[df['Label'] == 'Political']['Processed_Text'].dropna()
        nonpolitical_text = df[df['Label'] == 'NonPolitical']['Processed_Text'].dropna()
        print(f"Political memes: {len(political_text)}")
        print(f"Non-Political memes: {len(nonpolitical_text)}")

        political_words = []
        for text in political_text:
            political_words.extend(self.extract_words(text))

        nonpolitical_words = []
        for text in nonpolitical_text:
            nonpolitical_words.extend(self.extract_words(text))
            
        political_word_counts = Counter(political_words)
        nonpolitical_word_counts = Counter(nonpolitical_words)
        
        political_specific_words = []
        total_political = len(political_text)
        total_nonpolitical = len(nonpolitical_text)

        for word, pol_count in political_word_counts.items():
            nonpol_count = nonpolitical_word_counts.get(word, 0)
            if pol_count >= 3:
                ratio = pol_count / (nonpol_count + 1)
                pol_frequency = (pol_count / total_political) * 100
                nonpol_frequency = (nonpol_count / total_nonpolitical) * 100 if nonpol_count > 0 else 0
                
                political_specific_words.append({
                    'word': word,
                    'political_count': pol_count,
                    'nonpolitical_count': nonpol_count,
                    'ratio': ratio,
                    'total_count': pol_count + nonpol_count,
                    'political_frequency_%': round(pol_frequency, 2),
                    'nonpolitical_frequency_%': round(nonpol_frequency, 2)
                })

        political_df = pd.DataFrame(political_specific_words)
        if len(political_df) > 0:
            political_df = political_df.sort_values('ratio', ascending=False)
        
        political_df.to_csv(self.config.output_csv, index=False)
        print(f"Political-specific words saved to {self.config.output_csv}")
