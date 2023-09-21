#!/usr/bin/env python
# coding: utf-8

# # question 01
To find the probability that an employee is a smoker given that they use the health insurance plan, we can use conditional probability.

Let's denote:
- \(A\): Event that an employee uses the health insurance plan.
- \(B\): Event that an employee is a smoker.

We want to find \(P(B|A)\), which is the probability of being a smoker given that the employee uses the health insurance plan.

From the information provided, we have:

\(P(A) = 0.7\) (probability of using the health insurance plan)
\(P(B|A) = 0.4\) (probability of being a smoker given that they use the health insurance plan)

The conditional probability formula is:

\[P(B|A) = \frac{P(A \cap B)}{P(A)}\]

Here, \(P(A \cap B)\) is the probability of both using the health insurance plan and being a smoker.

Plugging in the values:

\[0.4 = \frac{P(A \cap B)}{0.7}\]

Solving for \(P(A \cap B)\):

\[P(A \cap B) = 0.4 \times 0.7 = 0.28\]

So, the probability that an employee is a smoker given that they use the health insurance plan is \(0.28\) or \(28\%\).
# # question 02
Bernoulli Naive Bayes and Multinomial Naive Bayes are two different variants of the Naive Bayes algorithm, which is commonly used in text classification and other tasks involving categorical features. Here are the key differences between them:

1. **Nature of Input Data**:

   - **Bernoulli Naive Bayes** is designed for binary/boolean features. It assumes that the features are binary-valued (i.e., they take on two values, typically 0 and 1).
   
   - **Multinomial Naive Bayes** is suitable for discrete features. It's commonly used for text classification where the features represent word counts or term frequencies.

2. **Feature Representation**:

   - **Bernoulli Naive Bayes** uses a binary feature representation. It considers whether a feature is present or absent in a sample.
   
   - **Multinomial Naive Bayes** typically uses integer feature counts. It's well-suited for data where each feature represents the frequency of occurrences (e.g., word counts).

3. **Application**:

   - **Bernoulli Naive Bayes** is commonly used in applications where binary features are important, such as document classification, sentiment analysis, and spam filtering.
   
   - **Multinomial Naive Bayes** is widely used in natural language processing tasks, particularly in text classification tasks like topic categorization, email filtering, and document classification.

4. **Probability Calculation**:

   - **Bernoulli Naive Bayes** calculates the likelihood of a feature occurring in each class as a binary event (present or absent).
   
   - **Multinomial Naive Bayes** calculates probabilities based on feature counts or frequencies. It considers how many times a particular feature occurs in each class.

5. **Model Representation**:

   - **Bernoulli Naive Bayes** uses a binary occurrence matrix where entries are 0 or 1.
   
   - **Multinomial Naive Bayes** typically uses integer counts (word frequencies) as the feature representation.

6. **Smoothing**:

   - Both variants may employ smoothing techniques (e.g., Laplace smoothing) to handle cases where certain features have zero counts in a particular class.

In summary, the choice between Bernoulli and Multinomial Naive Bayes depends on the nature of your data and the specific problem you are trying to solve. If you're dealing with binary features, Bernoulli Naive Bayes is more appropriate. For tasks involving counts or frequencies of features (common in text analysis), Multinomial Naive Bayes is often the better choice.
# # question 03
Bernoulli Naive Bayes does not handle missing values inherently. It assumes that each feature is binary, meaning it is either present or absent in each sample. If a feature's value is missing, it cannot be considered as either present (1) or absent (0), which conflicts with the assumption of binary features.

When dealing with missing values in a dataset that you intend to use with Bernoulli Naive Bayes, you have a few options:

1. **Imputation**:
   - You can apply an imputation technique to fill in the missing values with an appropriate estimate. For binary features, this might involve using the mode (most frequent value) of the feature.

2. **Data Transformation**:
   - You might choose to transform your data in a way that missing values are no longer an issue. For example, you could encode missing values as a special category (e.g., -1), effectively treating them as a third category.

3. **Omission**:
   - If the missing values are isolated instances and removing them won't significantly impact your dataset, you can choose to omit those samples.

4. **Special Handling**:
   - In some cases, you might have domain-specific knowledge that allows you to handle missing values in a specific way that makes sense for your problem.

It's important to note that whatever approach you choose, you should be consistent in applying it to both the training and testing sets. Additionally, carefully consider the implications of your chosen approach on the assumptions and performance of your Bernoulli Naive Bayes model.
# # question 04
Yes, Gaussian Naive Bayes can be used for multi-class classification tasks. Although it is named "Gaussian" because it assumes that the features are normally distributed, it can still be adapted for scenarios with categorical or discrete features.

In the context of multi-class classification, you would typically employ one of two strategies:

1. **One-Versus-Rest (OvR) or One-Versus-All (OvA)**:

   - In this approach, you train one binary classifier for each class, treating it as the positive class while treating all other classes as the negative class. For example, if you have three classes (A, B, and C), you would train three classifiers: A vs (B and C), B vs (A and C), and C vs (A and B).

   - At prediction time, you apply all classifiers and choose the class that yields the highest predicted probability.

2. **One-Versus-One (OvO)**:

   - In this approach, you train a binary classifier for each pair of classes. For N classes, this results in N(N-1)/2 classifiers. For example, in a three-class problem, you'd train three classifiers: A vs B, A vs C, and B vs C.

   - At prediction time, you apply all classifiers and use a voting scheme to determine the final class.

Gaussian Naive Bayes can be applied to both of these strategies. It works by assuming that the features are normally distributed within each class. This assumption might not always hold, especially if the features are not continuous or do not follow a Gaussian distribution. In practice, it can still work reasonably well for many types of data, but it's important to be aware of this underlying assumption and to consider the nature of your data when choosing a classifier.
# # question 05

# In[ ]:


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
url = "spambase.data"
data = pd.read_csv(url, header=None)

# Split data into features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Initialize classifiers
classifiers = {
    'Bernoulli Naive Bayes': BernoulliNB(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Gaussian Naive Bayes': GaussianNB()
}

# Evaluate classifiers using 10-fold cross-validation
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X, y, cv=10)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y, clf.fit(X, y).predict(X))
    precision = precision_score(y, clf.fit(X, y).predict(X))
    recall = recall_score(y, clf.fit(X, y).predict(X))
    f1 = f1_score(y, clf.fit(X, y).predict(X))
    
    print(f"Classifier: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")
    print("-" * 30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




