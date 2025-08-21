# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.ensemble import RandomForestClassifier  # For the classification model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For evaluating the model
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced plotting

# Step 1: Load the dataset
# Make sure to provide the correct path to your CSV file
data = pd.read_csv(r"C:\Users\anjoj\Downloads\Iris (1).csv")

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Step 2: Prepare the data
# Define the features (input data) and the target variable (what we want to predict)
X = data.iloc[:, 1:5]  # Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
y = data['Species']     # Target: Species

# Step 3: Split the dataset into training and testing sets
# This helps us evaluate the model's performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the model
# We will use a Random Forest Classifier for this task
model = RandomForestClassifier()

# Step 5: Train the model
# Fit the model to the training data
model.fit(X_train, y_train)

# Step 6: Make predictions
# Use the trained model to predict the species of the test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy:.2f}')

# Generate a confusion matrix to see how well the model performed
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Generate a classification report for more detailed evaluation
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Step 8: Visualize the confusion matrix
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data['Species'].unique(), 
            yticklabels=data['Species'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
