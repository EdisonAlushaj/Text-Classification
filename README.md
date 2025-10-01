# AI Internship Challenge - Text Classification Project

This project is a text classification model built for the LinkPlus AI Internship Challenge. The model is trained to categorize news articles into one of four topics: computer graphics, hockey, medicine, or Christian religion.

## Project Overview

The project follows a standard machine learning workflow:
1.  **Data Loading:** The "20 Newsgroups" dataset is loaded.
2.  **Exploratory Data Analysis (EDA):** The distribution of samples across categories is visualized.
3.  **Data Preprocessing:** The text data is cleaned by lowercasing, removing punctuation, and filtering out common English stopwords.
4.  **Feature Engineering:** The cleaned text is converted into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.
5.  **Model Training:** A Multinomial Naive Bayes classifier is trained on the vectorized text data.
6.  **Evaluation:** The model's performance is evaluated on a held-out test set using accuracy, a classification report, and a confusion matrix.
7.  **Prediction:** A function is provided to classify new, unseen text.

## Dataset

The project uses the **20 Newsgroups dataset**, which is publicly available through the `scikit-learn` library.
- **Source:** [Scikit-learn documentation](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
- **Categories Used:** `comp.graphics`, `rec.sport.hockey`, `sci.med`, `soc.religion.christian`

## How to Run the Project

1.  **Clone the repository.**
2.  **Navigate to the project directory.**
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the main script:**
    ```bash
    python text_classifier.py
    ```
    The script will run all steps, display two plots, and print the final evaluation and prediction results to the console.

## Results

The model achieved an accuracy of approximately **95%** on the test data. The detailed classification report and confusion matrix show that the model is highly effective at distinguishing between the four selected topics.
