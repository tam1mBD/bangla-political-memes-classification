# Bangla Political Memes Classification

## Project Description
A machine learning project for classifying Bangla political memes using text-based and vision-language approaches. The repository provides a reproducible, notebook-driven workflow for text extraction (EasyOCR), preprocessing, exploratory analysis, keyword mining, and model training (text-only and CLIP-based).

Project context:
- Project Name: Bangla Political Memes Classification
- Project Type: ML Project
- Purpose: Detect and classify political content in Bangla memes
- Target Users: Researchers, data scientists, content moderation teams
- Current Status: In Development

## Features
- End-to-end Jupyter Notebook workflow
- EasyOCR-based text extraction from meme images
- Text preprocessing for Bangla content
- Keyword mining to identify political terms
- Text-only classification model
- CLIP-based vision-language classification
- Basic evaluation and visualizations

## Technologies Used
- Python, Jupyter Notebook
- PyTorch, torchvision, CLIP/transformers
- scikit-learn, pandas, numpy
- OpenCV, Pillow
- EasyOCR
- matplotlib, seaborn

## Project Architecture / Workflow
1. Data collection and labeling (images + metadata)
2. Text extraction from images (EasyOCR): [text_extraction.ipynb](text_extraction.ipynb)
3. Text cleaning and normalization: [text_preprocessing.ipynb](text_preprocessing.ipynb)
4. Exploratory data analysis and visualization: [text_analysis.ipynb](text_analysis.ipynb)
5. Political keyword mining: [find_political_word.ipynb](find_political_word.ipynb)
6. Text-only classification model: [classification_model_using_text.ipynb](classification_model_using_text.ipynb)
7. CLIP-based image/text classification: [clip_model.ipynb](clip_model.ipynb)
8. Evaluation and reporting

## Installation Instructions
Prerequisites:
- Python 3.10+ and pip
- JupyterLab or Jupyter Notebook
- Optional: CUDA-enabled GPU for faster EasyOCR and model training

Using pip (recommended):
```sh
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows (Powershell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install jupyter numpy pandas scikit-learn torch torchvision torchaudio pillow opencv-python easyocr matplotlib seaborn ftfy regex tqdm sentencepiece transformers
```

## Usage Guide
1. Launch Jupyter:
```sh
jupyter lab
# or
jupyter notebook
```
2. Execute notebooks in this order:
   - [text_extraction.ipynb](text_extraction.ipynb)
   - [text_preprocessing.ipynb](text_preprocessing.ipynb)
   - [text_analysis.ipynb](text_analysis.ipynb)
   - [find_political_word.ipynb](find_political_word.ipynb)
   - [classification_model_using_text.ipynb](classification_model_using_text.ipynb)
   - [clip_model.ipynb](clip_model.ipynb)

3. Ensure dataset paths are correctly set in the first cells of each notebook.

## Configuration
- EasyOCR reader initialization (Bangla, optional English):
```python
from easyocr import Reader
reader = Reader(['bn', 'en'], gpu=True)  # set gpu=False if no CUDA
# Example usage:
# text_lines = reader.readtext('path/to/image.png', detail=0)
```
- Recommended directories (create as needed): `data/`, `outputs/`, `models/`, `reports/`.

## Folder Structure
- [.gitignore](.gitignore)
- [LICENSE](LICENSE)
- [README.md](README.md)
- [classification_model_using_text.ipynb](classification_model_using_text.ipynb)
- [clip_model.ipynb](clip_model.ipynb)
- [find_political_word.ipynb](find_political_word.ipynb)
- [text_analysis.ipynb](text_analysis.ipynb)
- [text_extraction.ipynb](text_extraction.ipynb)
- [text_preprocessing.ipynb](text_preprocessing.ipynb)



## Future Enhancements
- Unified training and inference scripts
- Multimodal fusion of text and image features
- Hyperparameter search and cross-validation
- Model export and lightweight inference API
- Dataset card and documentation
- Dockerfile and reproducible environment setup
- Automated tests and CI

## Contributing Guidelines
- Fork the repository and create a feature branch
- Follow clear commit messages and open a pull request with a concise description
- Include minimal examples or screenshots when relevant
- Ensure notebooks run end-to-end before submitting

## License
This project is licensed under the terms specified in [LICENSE](LICENSE).