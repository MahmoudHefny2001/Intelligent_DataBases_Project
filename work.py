import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt


# Data Cleaning

files = glob.glob('*.csv')

for file in files:
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(file)

    # Perform the data cleaning steps
    # Save the cleaned dataframe to a new CSV file
    cleaned_file = file.split('.')[0] + '_cleaned.csv'

    # df.to_csv(cleaned_file, index=False)


# Check for missing values in the entire dataframe
print(df.isnull().sum())

# Check for missing values in a specific column
# print(df['Store'].isnull().sum())
# print(df['Category'].isnull().sum())
# print(df['Date'].isnull().sum())
# print(df['Weekly_Sales'].isnull().sum())
# print(df['Holiday'].isnull().sum())


# Calculate summary statistics of a column
print(df['Store'].describe())
print(df['Category'].describe())
print(df['Date'].describe())
print(df['Weekly_Sales'].describe())
print(df['Holiday'].describe())

# Visualize the distribution of a column using a boxplot
sns.boxplot(df['Store'])
plt.show()

sns.boxplot(df['Category'])
plt.show()

# sns.boxplot(df['Date'])
# plt.show()

sns.boxplot(df['Weekly_Sales'])
plt.show()

sns.boxplot(df['Holiday'])
plt.show()

