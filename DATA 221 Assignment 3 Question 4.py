''' Question 4
Using the train-test data from Q3, train a KNN classifier.
Set the number of neighbors to k=5.
Train the model using the training data, then predict the labels of the test data.

After making predictions:
- Compute and display the confusion matrix
- Compute and print accuracy, precision, recall, and F1-score

Then write comments in your code explaining:
- What TP, TN, FP, FN mean in the context of kidney disease prediction
- Why accuracy alone may not be enough to evaluate a classification model
- Which metric is the most important if missing a kidney disease case is very serious, and why
'''

# Imports libraries and reads csv into a dataframe
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import numpy as np

kidney_disease_df = pd.read_csv("kidney_disease.csv")

# Convert binary categorical features to 1s and 0s and clean data so that it can be KNN imputed
kidney_disease_df = kidney_disease_df.replace({"notpresent": 0.0, 'present': 1.0,
                                               "normal": 1.0,"abnormal": 0.0,
                                               "yes":1.0, "no":0.0,
                                               "poor":0.0, "good":1.0,
                                               "ckd\t":1.0, "notckd\t":0.0,
                                               "ckd":1, "notckd":0,
                                               "\t":np.nan, "\tno":0,
                                               " yes":1, "\tyes":1,
                                               "\t?":np.nan})

# Fill missing data with KNN imputation
num_neighbors = 5
imputer = KNNImputer(n_neighbors=num_neighbors)
imputed_kidney_disease_df = pd.DataFrame(imputer.fit_transform(kidney_disease_df), columns=kidney_disease_df.columns)

# Creates feature matrix X of all columns except "classification" and saves "classification" as label vector Y
x = imputed_kidney_disease_df.drop("classification", axis=1)
y = imputed_kidney_disease_df["classification"]

# Splits training and testing data with a fixed seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=19)

# Trains the data and predicts labels for the test data
knn_model = KNeighborsClassifier(n_neighbors=num_neighbors, metric='euclidean')
trained_knn_model = knn_model.fit(x_train, y_train)
kidney_disease_prediction = trained_knn_model.predict(x_test)

# Measure accuracy using test labels and predicted labels

print(f"Confusion matrix: \n{confusion_matrix(kidney_disease_prediction, y_test)}")
print(f"")