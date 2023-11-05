import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
dataframe = pd.read_csv('financial_data_with_tickers_knn.csv')

# Separate the features and the target variable
X = dataframe[['P/E Ratio', 'D/E Ratio']]  # Features
y = dataframe['Trend']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the testing data
predictions = knn.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, predictions)}')
print(classification_report(y_test, predictions))

# Calculate the comparison
comparison = y_test == predictions

# Aggregate the data to count correct and incorrect predictions
correct_predictions = np.sum(comparison)
incorrect_predictions = len(comparison) - correct_predictions

# Calculate the accuracy
accuracy = correct_predictions / len(comparison)

# Data to plot
aggregated_data = {'Correct': correct_predictions, 'Incorrect': incorrect_predictions}

# Plot
fig, ax = plt.subplots()
bars = plt.bar(aggregated_data.keys(), aggregated_data.values(), color=['green', 'red'])

# Adding the text labels on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom')

ax.set_ylabel('Number of Predictions')
ax.set_title('Aggregated Comparison of Predictions')
ax.set_ylim(0, len(comparison) + 5)  # Set y limit higher to make room for text labels

# Display accuracy
plt.text(1, ax.get_ylim()[1]-5, f'Accuracy: {accuracy:.2f}', ha='center', va='bottom', fontsize=12, color='blue')

plt.show()
