import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv("data/titanic.csv")
# Show first rows
print(df.head())
# Dataset info
print(df.info())
# Statistical summary
print(df.describe())
# Missing values
print(df.isnull().sum())
# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)
# Histogram
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.savefig("images/histogram.png")
plt.show()
# Count Plot
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("images/countplot.png")
plt.show()
# Box Plot
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Passenger Class vs Age")
plt.savefig("images/boxplot.png")
plt.show()
# Correlation Heatmap
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png")
plt.show()
print("EDA Analysis Completed Successfully")