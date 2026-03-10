import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
url = "https://raw.githubusercontent.com/biolab/datasets/master/breast-cancer.csv"
df = pd.read_csv(url)
# (a) Print the first five rows
print("First five rows of the dataset:")
print(df.head())
# (b) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())
# (c) Columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)
# (d) Detect and handle null values
print("\nDetecting null values:")
print(df.isnull().sum())
for column in df.columns:
	mode_value = df[column].mode()[0]
	df[column].fillna(mode_value, inplace=True)
print("\nNull values after handling:")
print(df.isnull().sum())
# (e) Split the data into train and test sets

X = df.drop('Class', axis=1) # Features
y = df['Class'] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the performance of the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))