''' Question 2
Using crime.csv and the ViolentCrimesPerPop column, create two plots using matplotlib.
First, create a histogram that shows how the values are distributed.
Second, create a box plot for the same data.
Each plot must include a clear title and axis labels.

After generating the plots, write comments in your code describing:
- What the histogram shows about how the data values are spread
- What the box plot shows about the median
- Whether the box plot suggests the presence of outliers
'''

# Imports libraries and reads csv as a DataFrame
import pandas as pd
import matplotlib.pyplot as plt
crime_df = pd.read_csv("crime1.csv")

# Creates histogram
plt.hist(crime_df["ViolentCrimesPerPop"], edgecolor='white')
plt.xlabel("Proportion of Violent Crimes")
plt.ylabel("Frequency")
plt.title("Distribution of Proportion of Violent Crime per Population")
plt.show()

# Creates box plot
plt.boxplot(crime_df["ViolentCrimesPerPop"])
plt.xlabel("All Populations")
plt.ylabel("Proportion of Violent Crimes")
plt.title("Box Plot of Proportion of Violent Crime per Population")
plt.show()

'''
The histogram shows that the distribution is slightly right-skewed. 
Most of the values are concentrated to left (lower crime rates).
The boxplot shows that the median is slightly below 0.4.
The boxplot does not show any outliers, but it shows a longer tail on the right.
The boxplot and the histogram support each other and what we found in Q1: the distribution of violent crime rates is right-skewed.
'''