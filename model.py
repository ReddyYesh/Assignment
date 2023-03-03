import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('C:/Users/paipa/OneDrive/Desktop/aug_train.csv')
df


labels = {
    0:" Not looking for job change",
    1:"looking for job change",
    
}
df.shape
df.dtypes
df.shape[0]
df.shape[1]
df.isnull()
df.isnull().sum()


X = df[['enrollee_id']] 
y = df['target'] 
X = pd.get_dummies(X) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# predict on new data
new_data = pd.read_csv("C:/Users/paipa/OneDrive/Desktop/aug_test.csv")
X_new = pd.get_dummies(new_data[['enrollee_id']])
y_pred_new = rf.predict(X_new)
print('Predictions for new data:', y_pred_new)

submission = pd.DataFrame({'enrollee_id':X_new['enrollee_id'],'target':y_pred_new})
submission.head(10)

submission.to_csv('submissions.csv',index=False)

#export the model
pickle.dump(rf,open('model.pkl','wb'))

#load the model and test with a custom input
model = pickle.load(open('model.pkl','rb'))

x = (21651,33241)
predict = model.predict(X_new)
print(predict)
print("Hey")
print(labels[predict[0]])
if (predict[0] == 0):
  print('not Looking for a job')
else:
  print(' is Looking for a job')


