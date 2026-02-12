''' Question 1
Load crime.csv into a pandas DataFrame. Focus on ViolentCrimesPerPop. Compute and print the following:
- Mean
- Median
- Standard dev
- Minimum value
- Maximum value

Then, write comments answering:
- Compare the mean and median. Does the distribution look symmetric or skewed? Explain.
- If there are extreme values, which statistic is more affected: mean or median? Explain.
'''

import pandas as pd

crime_df = pd.read_csv("crime1.csv")

print("Violent Crime per Pop Summary Statistics:")
print(f"Mean: {crime_df["ViolentCrimesPerPop"].mean()}")
print(f"Median: {crime_df["ViolentCrimesPerPop"].median()}")
print(f"Standard deviation: {crime_df["ViolentCrimesPerPop"].std()}")
print(f"Minimum: {crime_df["ViolentCrimesPerPop"].min()}")
print(f"Maximum: {crime_df["ViolentCrimesPerPop"].max()}")