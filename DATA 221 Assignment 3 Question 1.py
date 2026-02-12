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

# Reads the csv into a DataFrame
import pandas as pd
crime_df = pd.read_csv("crime1.csv")

# Finds summary statistics using methods
print("Violent Crime per Pop Summary Statistics:")
print(f"Mean: {crime_df["ViolentCrimesPerPop"].mean()}")
print(f"Median: {crime_df["ViolentCrimesPerPop"].median()}")
print(f"Standard deviation: {crime_df["ViolentCrimesPerPop"].std()}")
print(f"Minimum: {crime_df["ViolentCrimesPerPop"].min()}")
print(f"Maximum: {crime_df["ViolentCrimesPerPop"].max()}")

'''
The mean is 0.44119, and the median is 0.39.
Since the mean is greater than the median, the distribution is right-skewed.
The upper 50% of the values are pulling the mean towards the right.
'''

'''
Extreme values affect the mean more than the median.
Mean is affected by the value of each data point. Median is affected by the order of values.
Consider: 1, 100, 101, 102, 103 and 99, 100, 101, 102, 103
The mean is heavily affected by the change from 1 to 99 (81.4 vs 101).
The median is exact same for both (101) because the middle value is the exact same.
'''