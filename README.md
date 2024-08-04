# Multi-Class Text Classification using SVM

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Definition](#problem-definition)
3. [Objectives](#objectives)
4. [Limitations](#limitations)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Methodology](#methodology)
9. [Dataset Description](#dataset-description)
10. [Preprocessing Techniques](#preprocessing-techniques)
11. [Model Development](#model-development)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Deployment and Results](#deployment-and-results)
14. [Conclusion](#conclusion)
15. [Future Scope](#future-scope)

## Introduction
Support Vector Machine (SVM) was initially designed for binary classification. This project explores a novel framework called Class-Incremental Learning (CIL) for multi-class SVM. CIL consists of incremental feature selection and incremental training to update SVM classifiers' knowledge in text classification when new classes are added.

## Problem Definition
This project aims to construct an advanced news article classification system using SVM for precise categorization of news articles across multiple domains, including technology, sports, politics, entertainment, and business. The primary objective is to design and implement a robust text classification pipeline with intricate preprocessing steps.

## Objectives
- Develop a sophisticated text classification system using SVM algorithms.
- Process raw textual data with techniques like tokenization, stemming, and vectorization.
- Construct a robust classification pipeline that optimizes SVM model performance.
- Develop an intuitive user interface for instant category prediction of new articles.

## Limitations
- Computational intensity
- Sensitivity to noise
- Difficulty with large feature spaces
- Binary classification nature

## Installation
To install and set up the Multi-Class Text Classification application, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/multi-class-text-classification.git
   ```
2. Navigate to the project directory:
   ```sh
   cd multi-class-text-classification
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   jupyter notebook
   ```
5. Open the `visualstudio tc code.ipynb` notebook in Jupyter.

## Usage
1. Open the Jupyter notebook file `visualstudio tc code.ipynb`.
2. Follow the instructions in the notebook to preprocess data, train the model, and evaluate its performance.

## Project Structure
- `visualstudio tc code.ipynb`: Jupyter notebook containing the entire code for data preprocessing, model training, evaluation, and deployment.

## Methodology
### Text Preprocessing
- Lowercasing
- Tokenization
- Stemming/Lemmatization
- Removing stopwords
- TF-IDF Vectorization

### Model Development
- SVM algorithm for classification
- Randomized hyperparameter search using `RandomizedSearchCV`
- Evaluation metrics: accuracy, precision, recall, F1-score

## Dataset Description
The dataset consists of news articles from various categories: business, entertainment, politics, sport, and tech. It is sourced from Kaggle and contains labeled articles for classification.

## Preprocessing Techniques
- Lowercasing
- Tokenization
- Stemming/Lemmatization
- Removing stopwords
- TF-IDF Vectorization

## Model Development
The SVM algorithm is used for classification, trained on the preprocessed data, and fine-tuned using hyperparameter search.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## Deployment and Results
The deployment phase involves practical implementation, scalability, and integration into existing systems. The model's performance is evaluated on unseen test data using various metrics.

## Conclusion
The project successfully employed SVM to categorize BBC news articles into predefined topics with high accuracy. The model demonstrated commendable performance metrics, positioning it as a valuable tool for news categorization.

## Future Scope
- Implement advanced NLP techniques like embedding models (Word2Vec, GloVe) or transformer models (BERT).
- Adapt the model for real-time data streaming.
- Enhance model explainability and interpretability.
- Explore ensemble learning techniques.
