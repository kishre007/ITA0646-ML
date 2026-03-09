# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Step 1: Load the Iris dataset
data = load_iris()
iris = pd.DataFrame(data.data, columns=data.feature_names)
iris['species'] = data.target

# Plot graph
plt.figure(figsize=(10,6))
plt.scatter(iris['sepal width (cm)'], iris['sepal length (cm)'], c=iris['species'], s=50)
plt.title('Sepal Width vs Sepal Length')
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.show()

# Prepare data
X = iris.drop('species', axis=1)
y = iris['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict new data
new_data = [[5,3,1,0.3]]
predicted_species = model.predict(new_data)

print("Predicted class:", predicted_species[0])