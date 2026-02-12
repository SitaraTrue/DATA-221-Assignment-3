''' Question 3
Load kidney_disease.csv into a pandas DataFrame.
Create a feature matrix X that contains all columns except classification.
Create a label vector y using the classification column.
Then, split the dataset into training (70%) and testing (30%) data.
Use train_test_split with a fixed random_state.

After performing the split, write comments in your code explaining:
- Why we should not train and test a model on the same data
- What the purpose of the testing set is
'''

# Imports libraries
from sklearn.model_selection import train_test_split
import pandas as pd

# Reads csv into a dataframe
kidney_disease_df = pd.read_csv("kidney_disease.csv")

# Creates feature matrix X of all columns except "classification" and saves "classification" as label vector Y
x = kidney_disease_df.drop("classification", axis=1)
y = kidney_disease_df["classification"]

# Splits training and testing data with a fixed seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=19)

'''
The purpose of testing data is to see how the model performs with unfamiliar data.
If you test and train on the same data, the model will be overfitted to the training data.
The model learned the training data, so testing it on the same data won't show how accurate it is with new data.
'''