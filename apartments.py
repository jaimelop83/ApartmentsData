import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('apartments_for_rent_classified_100K.csv', delimiter=';', encoding='cp1252 ', low_memory=False)

print(df.head())

# Plotting the distribution of the prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=100, kde=True)   # kde=True adds a kernel density estimate
plt.xlim(0, df['price'].quantile(0.99)) # Limit the x-axis to the 99th percentile
plt.ylim(0, 50000) # Limit the y-axis to 1000
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

numeric_df = df.select_dtypes(include=['float64', 'int64']) # Select only the numeric columns

corr = numeric_df.corr() # Calculate the correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f") # Plot the correlation matrix
plt.title('Correlation Matrix')
plt.show()

state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']

plt.figure(figsize=(10, 6))
sns.barplot(x='state', y='count', data=state_counts, hue='state', palette='viridis')
plt.title('Number of Apartments by State')
plt.xlabel('State')
plt.ylabel('Number of Apartments')
plt.xticks(rotation=45) # Rotate the state names for better readability
plt.legend([],[], frameon=False) # Remove the legend
plt.show()

# Calculate the average price per state
state_avg_price = df.groupby('state')['price'].mean().reset_index()
state_avg_price.columns = ['state', 'average_price']

# Sort the states by average price in descending order
state_avg_price = state_avg_price.sort_values('average_price', ascending=False)

# Plot the top N states with the most expensive apartments
top_n = 10 # Change this to plot more or fewer states
plt.figure(figsize=(10, 6))
sns.barplot(x='state', y='average_price', data=state_avg_price.head(top_n), palette='magma')

plt.title(f'Top {top_n} States with the Most Expensive Apartments')
plt.xlabel('State')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45) # Rotate the state names for better readability
plt.show()