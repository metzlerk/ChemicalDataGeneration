from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import importlib
import functions as f
# Reload the functions module after updates
importlib.reload(f)

data = pd.read_feather('data/test_data.feather')

# Assuming the last column is the target variable
X = data.iloc[:, 2:-9]
y = data.iloc[:, -9]

# Split the data into training and testing sets
# making it so that most data goes in test and then reducing size of test. just a lazy way of reducing total data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
_, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.1, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

