#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import csv

def fix_csv(file_path):
    corrected_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)
        corrected_data.append(headers)
        for row in reader:
            password = ','.join(row[:-1])
            strength = row[-1]
            corrected_data.append([password, strength])
    return corrected_data

file_path = '/Users/JPL/Downloads/Project/data.csv'

corrected_lines = fix_csv(file_path)
buffer = io.StringIO()
csv.writer(buffer).writerows(corrected_lines)
buffer.seek(0)
data = pd.read_csv(buffer, header=0)

X = data['password'].fillna('')
y = data['strength'].astype(int)

vectorizer = TfidfVectorizer(analyzer='char', lowercase=False)
X = vectorizer.fit_transform(X)

# KFold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results of each fold
accuracies = []
confusion_matrices = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))
    confusion_matrices.append(confusion_matrix(y_test, predictions))

# Print overall results
print(f"Average Accuracy: {np.mean(accuracies):.2f}")
print("\nClassification Report for last fold: ")
print(classification_report(y_test, predictions))

# Visualize the confusion matrix for the last fold
cm = confusion_matrices[-1]
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=clf.classes_,
            yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Last Fold')
plt.show()
