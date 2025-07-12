import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
 
df = pd.read_csv("student_data.csv")
 
df['Result'] = df['Result'].map({'Pass': 1, 'Fail': 0})

X = df[['Attendance', 'StudyHours', 'InternalMarks', 'PreviousGrade']]
y = df['Result']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
model = LogisticRegression()
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
 
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm, cmap='coolwarm')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
plt.show()
