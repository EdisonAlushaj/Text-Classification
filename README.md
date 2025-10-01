# AI Internship Challenge - Text Classification

This project is a text classification model built to categorize news articles into one of four topics: hockey, medicine, religion, or computer graphics.

## Dataset

The project uses the "20 Newsgroups" dataset, which is publicly available through the `scikit-learn` library.
- **Source:** [Scikit-learn documentation](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
- **Categories Used:** `rec.sport.hockey`, `sci.med`, `soc.religion.christian`, `comp.graphics`

## How to Run

1.  Clone the repository.
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main script:
    ```bash
    python text_classifier.py
    ```

## Results

The Naive Bayes model achieved an accuracy of approximately 95% on the test set. The confusion matrix shows that the model performs very well, with most errors occurring between the 'comp.graphics' and 'rec.sport.hockey' categories.
