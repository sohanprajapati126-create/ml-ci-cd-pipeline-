import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

#Load Dataset
data = pd.read_csv('data/iris.csv')

#Preprocess the dataset
X = data.drop('species', axis = 1)
y = data['species']

#Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train RandomForest model
model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(X_train, y_train)

#Save the model
joblib.dump(model, 'model/iris_model.pkl')

#Make Predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')