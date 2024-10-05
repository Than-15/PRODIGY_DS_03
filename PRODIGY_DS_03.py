import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

zip_file = r'C:\Users\User\Downloads\bank+marketing.zip'

with zipfile.ZipFile(zip_file, 'r') as z:
    print(z.namelist())

    with z.open('bank.zip') as inner_zip_file:
        with zipfile.ZipFile(inner_zip_file) as inner_zip:
            print(inner_zip.namelist())

            with inner_zip.open('bank-full.csv') as f:
                data = pd.read_csv(f, delimiter=';')

print("Initial Data:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

data_encoded = pd.get_dummies(data, drop_first=True)

print("\nEncoded Data:")
print(data_encoded.head())

X = data_encoded.drop('y_yes', axis=1)
y = data_encoded['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(15, 10))
tree.plot_tree(classifier, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()
