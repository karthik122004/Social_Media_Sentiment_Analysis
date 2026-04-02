# Social Media Sentiment Analysis

This project classifies synthetic social media comments into three sentiment classes — positive, negative, and neutral — and explores basic patterns in the data using Python, pandas, and scikit-learn.

## Goal

Demonstrate an end-to-end text classification workflow:  
data loading, exploratory data analysis (EDA), text preprocessing, feature engineering with TF-IDF, model training, and evaluation.

## Dataset

The dataset contains short social media-style comments with fine-grained emotion labels (e.g. *Joy*, *Fear*, *Admiration*). For this project, these emotions are grouped into three broader sentiment classes:

- **positive** (e.g. Positive, Happiness, Joy, Love, Amusement, Enjoyment, Admiration, Affection, Awe, Acceptance, Adoration, Anticipation)  
- **negative** (e.g. Negative, Anger, Fear, Sadness, Disgust, Disappointed)  
- **neutral** (e.g. Neutral, Surprise)  

Rows that do not map to these groups are removed to ensure enough examples per class.

## Methodology

1. **EDA**  
   - Inspect sentiment class distribution before and after grouping.  
   - Analyze comment length distribution.

2. **Text Preprocessing**  
   - Lowercasing, removal of URLs, mentions, hashtags, emojis, and non-alphabetic characters.  
   - Creation of a cleaned text field.

3. **Feature Engineering**  
   - TF-IDF vectorization of the cleaned text (`max_features=5000`, English stopwords).

4. **Modeling**  
   - **Logistic Regression** baseline.  
   - **Linear SVM (LinearSVC)** as a second baseline.

5. **Evaluation**  
   - Train/test split with stratification.  
   - Accuracy and classification report (precision, recall, F1-score) per class.  
   - Confusion matrix visualization for the best model.

## Results

- Both Logistic Regression and Linear SVM reach around **0.77** accuracy on the test set.  
- Linear SVM provides slightly more balanced precision/recall across classes, especially for the negative class.  
- The Linear SVM model is selected as the final model and used for the confusion matrix and example predictions.

## Limitations & Next Steps

- The dataset is synthetic and the grouped labels are still somewhat imbalanced, which hurts performance on neutral and negative comments.  
- Possible next steps include collecting more real-world data, applying techniques to handle class imbalance (e.g. class weights), and experimenting with more advanced models such as pretrained transformer-based text embeddings.

## Tools

- Python, pandas, NumPy  
- scikit-learn (TF-IDF, Logistic Regression, LinearSVC, metrics)  
- matplotlib, seaborn

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/karthik122004/Social_Media_Sentiment_Analysis.git
   cd Social_Media_Sentiment_Analysis
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook
   ```
   Then open `Sentiment_Analysis.ipynb` from the browser interface.
   
## Data

This project uses the **Social Media Sentiments Analysis Dataset** by Kashish Parmar on Kaggle:
https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset

The raw CSV file is not redistributed in this repository.  
To run the notebook, download the dataset from Kaggle and save it as `sentimentdataset.csv` in the project folder (or update the path in the notebook).
