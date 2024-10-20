import numpy as np
from sklearn.datasets import load_iris
##from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


iris=load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy*100:.2f}%')
print("Classfication report:")
print(classification_report(y_pred))



sample_flower=X_test[0].reshape(1, -1)

predicted_class=model.predict(sample_flower)

target_names=iris.target_names

print("Predicted flower name:", target_names(predicted_class[0])



