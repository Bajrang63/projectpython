Fake News Detection Project
Table of Contents
Project Overview
Features
Dataset
Technologies Used
Setup and Installation
Usage
Model Performance
API Structure (Optional)
Future Enhancements
Project Overview
This project develops a machine learning-based system to detect fake news articles. It involves loading and preprocessing text data, extracting features using TF-IDF, training and evaluating classification models (Logistic Regression and RandomForestClassifier), and providing a function for real-time news prediction. The goal is to combat misinformation by automatically classifying news as 'Real' or 'Fake'.

Features
Data Loading & Combination: Loads news articles from Fake.csv and True.csv and combines them into a single dataset.
Text Preprocessing: Cleans and prepares text data by converting to lowercase, removing punctuation, special characters, numbers, stop words, and applying stemming.
Feature Extraction: Transforms text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
Model Training: Trains two robust classification models: Logistic Regression and RandomForestClassifier.
Model Evaluation: Assesses model performance using accuracy, classification reports (precision, recall, F1-score), and confusion matrices.
Prediction Function: Provides a utility to predict the authenticity of new, unseen news articles.
Model Persistence: Saves the trained models and vectorizer for future use without retraining.
Dataset
The project uses two CSV files:

Fake.csv: Contains news articles labeled as 'fake'.
True.csv: Contains news articles labeled as 'real'.
The combined dataset consists of 3221 news articles after handling malformed entries.

Technologies Used
Python 3.x
pandas (for data manipulation)
scikit-learn (for machine learning models and utilities)
nltk (for natural language processing tasks like tokenization, stopwords, and stemming)
matplotlib (for plotting and visualization)
seaborn (for enhanced data visualizations)
joblib (for model persistence)
Flask (for API structure - optional, if deploying)
Setup and Installation
To set up the project locally, follow these steps:

Clone the repository:

git clone <your-repository-url>
cd fake-news-detection
Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:

pip install pandas scikit-learn nltk matplotlib seaborn joblib Flask
Download NLTK stopwords: Open a Python interpreter or add to your script:

import nltk
nltk.download('stopwords')
Place Data Files: Ensure Fake.csv and True.csv are in the project's root directory or update the loading paths in the code.

Usage
Run the Jupyter Notebook/Colab Notebook: Execute the cells sequentially to perform data loading, preprocessing, feature extraction, model training, and evaluation.

Predicting new news articles: Use the predict_news function defined in the notebook:

# Assuming 'model' and 'vectorizer' are loaded and 'preprocess_text' is defined
# Example:
# from your_script import predict_news

sample_news = """BREAKING NEWS: Government announces groundbreaking new policy to eliminate all taxes by next month, promising immediate prosperity for all citizens. Experts are baffled by the sudden announcement."""
prediction = predict_news(sample_news)
print(f"Prediction: {prediction}")
Model Performance
Both Logistic Regression and RandomForestClassifier models demonstrated excellent performance. The dataset was split into 80% training and 20% testing (645 test samples).

Comparison Table
Metric	Logistic Regression	RandomForestClassifier
Accuracy	0.9938	0.9953
Fake Precision	0.99	0.99
Fake Recall	0.99	1.00
Fake F1-Score	0.99	1.00
Real Precision	0.99	1.00
Real Recall	0.99	0.99
Real F1-Score	0.99	0.99
Confusion Matrices
Logistic Regression:

[[345   2]
 [  2 296]]
Errors: 2 False Positives (real classified as fake), 2 False Negatives (fake classified as real). Total 4 errors.
RandomForestClassifier:

[[347   0]
 [  3 295]]
Errors: 0 False Positives (real classified as fake), 3 False Negatives (fake classified as real). Total 3 errors.
Insights:

Both models achieved very high accuracy and F1-scores, indicating strong predictive power.
The RandomForestClassifier showed a slight edge in overall accuracy and achieved perfect recall for the 'fake' class, meaning it correctly identified all fake news articles in the test set. However, it had a slightly higher number of false negatives for the 'real' class compared to Logistic Regression.
The robust preprocessing and TF-IDF feature extraction proved highly effective for this task.
