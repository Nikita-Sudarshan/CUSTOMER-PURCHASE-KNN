import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("D:\PYTHON\MLMATHS\CUSTOMERKNN\data\data.csv")

X=df[['Age', 'EstimatedSalary']]
y=df['Purchased']

scaler= StandardScaler()
X= scaler.fit_transform(X)

X_train, X_test, y_train, y_test= train_test_split(
    X,y, test_size=0.2, random_state=42
)

model= KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("confusion matrix: \n", confusion_matrix(y_test, predictions))

k_values= range(1,21)
accuracies = []

for k in k_values:
    model= KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds= model.predict(X_test)
    accuracies.append(accuracy_score(y_test, preds))

plt.plot(k_values, accuracies)
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.show()