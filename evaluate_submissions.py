import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

# Load true labels
true_labels = pd.read_csv('Test_with_labels.csv')
print(f"True labels loaded: {len(true_labels)} samples")
print(f"Distribution: {true_labels['Label'].value_counts()}\n")

# List of submission files
submissions = {
    'Logistic Regression': 'submissions/submission_logistic_regression.csv',
    'Neural Network': 'submissions/submission_neural_network.csv',
    'CLIP': 'submissions/submission_clip.csv',
    'Ensemble': 'submissions/submission_ensemble.csv',
    'Late Fusion': 'submissions/submission_late_fusion.csv',
    'political_text': 'submissions/submission_hybrid_text_political.csv'
}

results = []

print("="*80)
print("EVALUATING ALL SUBMISSIONS")
print("="*80)

for model_name, file_path in submissions.items():
    # Load predictions
    predictions = pd.read_csv(file_path)
    
    # Merge with true labels
    merged = true_labels.merge(predictions, on='Image_name', suffixes=('_true', '_pred'))
    
    # Convert to binary
    y_true = (merged['Label_true'] == 'Political').astype(int)
    y_pred = (merged['Label_pred'] == 'Political').astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Prediction distribution
    pred_dist = predictions['Label'].value_counts()
    
    results.append({
        'Model': model_name,
        'Accuracy': acc * 100,
        'F1 Score': f1,
        'Political Predictions': pred_dist.get('Political', 0),
        'NonPolitical Predictions': pred_dist.get('NonPolitical', 0)
    })
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Predictions - Political: {pred_dist.get('Political', 0)}, NonPolitical: {pred_dist.get('NonPolitical', 0)}")
    
    # Show confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Check true label distribution
true_political = (true_labels['Label'] == 'Political').sum()
true_nonpolitical = (true_labels['Label'] == 'NonPolitical').sum()
print(f"\nTrue Label Distribution:")
print(f"  Political: {true_political} ({true_political/len(true_labels)*100:.1f}%)")
print(f"  NonPolitical: {true_nonpolitical} ({true_nonpolitical/len(true_labels)*100:.1f}%)")

print("\n" + "="*80)
