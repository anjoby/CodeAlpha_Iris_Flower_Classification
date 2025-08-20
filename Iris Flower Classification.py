
import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.ensemble import RandomForestClassifier  # For the classification model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For evaluating the model
import matplotlib.pyplot as plt  
import seaborn as sns 


data = pd.read_csv(r"C:\Users\anjoj\Downloads\Iris (1).csv")

print("First few rows of the dataset:")
print(data.head())


X = data.iloc[:, 1:5]  
y = data['Species']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data['Species'].unique(), 
            yticklabels=data['Species'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
