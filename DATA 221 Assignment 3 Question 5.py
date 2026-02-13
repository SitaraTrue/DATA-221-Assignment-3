''' Question 5
Using the same train-test data, train multiple knn models using different values of k. Train for:
    k = [1,3,5,7,9]
For each value of k, compute the test accuracy and store the results.
Create a small table showing each value of k and its corresponding accuracy.

Write comments explaining:
- How changing k affects the behavior of the model
- Why very small values of k may cause overfitting
- Why very large values of k may cause underfitting
'''

# Imports libraries and reads csv into a dataframe
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

kidney_disease_df = pd.read_csv("kidney_disease.csv")

# Convert binary categorical features to 1s and 0s and clean data so that it can be KNN imputed (fill missing values)
kidney_disease_df = kidney_disease_df.replace({"notpresent": 0.0, 'present': 1.0,
                                               "normal": 1.0,"abnormal": 0.0,
                                               "yes":1.0, "no":0.0,
                                               "poor":0.0, "good":1.0,
                                               "ckd\t":1.0, "notckd\t":0.0,
                                               "ckd":1, "notckd":0,
                                               "\t":np.nan, "\tno":0,
                                               " yes":1, "\tyes":1,
                                               "\t?":np.nan})

values_of_k = [1,3,5,7,9]
accuracy_list = []

# Loops through the different values of k
for num_neighbors in values_of_k:
    # Fills missing data with KNN imputation
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

    # Calculates accuracy and adds it to a list
    accuracy = accuracy_score(y_test, kidney_disease_prediction)
    accuracy_list.append(accuracy)

# Creates table mapping k-value to its accuracy
accuracy_df = pd.DataFrame({"Value of K": values_of_k, "Accuracy": accuracy_list})
print(accuracy_df)

# Finds the greatest accuracy and its corresponding k-value
greatest_accuracy = 0
for i in range(0, len(accuracy_list)):
    if accuracy_list[i] > greatest_accuracy:
        greatest_accuracy = accuracy_list[i]
        k_value_greatest_accuracy = values_of_k[i]

print(f"The highest test accuracy was {greatest_accuracy} using a k-value of {k_value_greatest_accuracy}.")

'''
Changing the value of k changes how many data points will be used to make predictions, therefore changing the accuracy.
Small values of k cause overfitting because the predictions become very sensitive to any small variation in the training data.
Large values of k cause underfitting because the predictions may overlook some variation in the data.
'''