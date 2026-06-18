import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

iris_raw = load_iris()

X = iris_raw.data
y = iris_raw.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, shuffle=True)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nFINAL PERFORMANNCE\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report(Includes Precision, Recall, &F1-Score):")
print(classification_report(y_test, y_pred, target_names=iris_raw.target_names))