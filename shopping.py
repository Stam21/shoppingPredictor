import csv
import numpy as np

import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    
    data = pd.read_csv(filename)
    evidence = []
    labels = []

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to Month and VisitorType columns
    data['Month'] = label_encoder.fit_transform(data['Month'])
    data['VisitorType'] = label_encoder.fit_transform(data['VisitorType'])

    # Convert Weekend column to integer (True -> 1, False -> 0)
    data['Weekend'] = data['Weekend'].astype(int)
    data['Revenue'] = data['Revenue'].astype(int)

    for i in data.index:
        evidence.append(data.loc[i, data.columns != "Revenue"].values.tolist())
        labels.append(data.loc[i, data.columns == "Revenue"].values.tolist()[0])
    return (evidence,labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    knn = KNeighborsClassifier(n_neighbors=1)  # Set n_neighbors=1 for k=1
    knn.fit(evidence, labels)
    
    return knn


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = 0
    true_negatives = 0
    total_positives = sum(labels)
    total_negatives = len(labels) - total_positives

    # Count true positives and true negatives
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1

    # Calculate the percentage of true positives and true negatives
    sensitivity = true_positives / total_positives if total_positives > 0 else 0
    specificity = true_negatives / total_negatives if total_negatives > 0 else 0

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
