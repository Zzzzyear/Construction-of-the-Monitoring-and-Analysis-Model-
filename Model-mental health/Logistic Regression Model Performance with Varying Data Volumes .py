import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

# Load the dataset.
file_path = r'C:\Users\ZQN\Desktop\GHQ12.csv'
data = pd.read_csv(file_path)

# Extract features and labels.
X = data.iloc[:, 2:15]
y = data.iloc[:, 15]

# Initialize the logistic regression model.
model = LogisticRegression()

# Define the number of data entries.
data_sizes = list(range(50, 401, 50))

# Store the average accuracy for each iteration.
average_accuracies = []

# Perform the loop operation eight times.
for data_size in data_sizes:
    # Extract the first {data_size} records.
    X_subset = X[:data_size]
    y_subset = y[:data_size]

    # Initialize ten-fold cross-validation.
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform ten-fold cross-validation and calculate the accuracy.
    cv_scores = cross_val_score(model, X_subset, y_subset, cv=kf)

    # Calculate the average accuracy.
    average_accuracy = cv_scores.mean()
    average_accuracies.append(average_accuracy)

# Plot the line chart.
plt.plot(data_sizes, average_accuracies, marker='o', label='Actual Accuracy')

# Polynomial fitting.
degree = 3  # The degree of the polynomial can be adjusted based on your needs.
coefficients = np.polyfit(data_sizes, average_accuracies, degree)
poly = np.poly1d(coefficients)
fit_values = poly(data_sizes)

# Draw the fitting curve.
plt.plot(data_sizes, fit_values, label=f'Fitted curves (Degree={degree})', linestyle='--')

plt.xlabel('Number of Data Records')
plt.ylabel('Average Accuracy')
plt.title('Logistic Regression Model Performance with Varying Data Volumes')
plt.legend()
plt.grid(True)
plt.show()
