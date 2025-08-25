import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

#load dataset
data = pd.read_csv('data/iris.csv')

#preprocess

X = data.drop('species', axis=1)
y = data['species']

#split into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#load the saved model 
model = joblib.load('model/iris_model.pkl')

#make the prediction 
y_pred = model.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')