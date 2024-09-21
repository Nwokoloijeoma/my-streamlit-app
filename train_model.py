import pandas as pd
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score, 
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils import resample
import plotly.express as px

# Load the dataset
file_path = "C:/Users/HP/OneDrive - Edge Hill University/Edgehill/Research Project/Report/predict+students+dropout+and+academic+success (4)/data.csv"
dataset = pd.read_csv(file_path, sep=';')

dataset['Target'].value_counts()
dataset['Target'] = LabelEncoder().fit_transform(dataset['Target'])
dataset['Target'].value_counts()

#As this prediction is whether a student will dropout or not, the number of "Enrolled" student is irrelevant. 
#The prediction is whether a student graduated or droped out. 
#The "Enrolled" values would be removed while proceeding with "Graduate" & "Dropout" values
dataset.drop(dataset[dataset['Target'] == 1].index, inplace = True)
dataset

dataset['Dropout'] = dataset['Target'].apply(lambda x: 1 if x==0 else 0)
dataset

#Standard Scaling the data
x = dataset.iloc[:, :36].values
#x = dataset[["Tuition fees up to date","Curricular units 1st sem (approved)","Curricular units 1st sem (grade)","Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)"]].values
print(x)
x = StandardScaler().fit_transform(x)
x

y = dataset['Dropout'].values
y

# Check distribution of the target variable
print("Class distribution in the target variable:")
print(dataset['Dropout'].value_counts())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Function to measure performance
def perform(y_pred):
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Dropout', 'Dropout']).plot()

# Define base classifiers and meta-classifier for stacking
base_models = [
    ('random_forest', RandomForestClassifier(n_estimators=500, criterion='entropy')),
    ('svc', SVC(C=0.1, kernel='linear', probability=True))
]
meta_model = LogisticRegression()

# Create stacking model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Cross-validation on the training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(stacking_model, x_train, y_train, cv=cv)

# Evaluate performance during cross-validation
accuracy_cv = accuracy_score(y_train, y_pred_cv)
precision_cv = precision_score(y_train, y_pred_cv, average='macro')
recall_cv = recall_score(y_train, y_pred_cv, average='macro')
f1_cv = f1_score(y_train, y_pred_cv, average='macro')

print("Cross-Validation Accuracy:", accuracy_cv)
print("Cross-Validation Precision:", precision_cv)
print("Cross-Validation Recall:", recall_cv)
print("Cross-Validation F1 Score:", f1_cv)

# Fit the stacking model on the entire balanced training data
stacking_model.fit(x_train, y_train)

# Save the trained model to a file
joblib.dump(stacking_model, 'student_risk_model.pkl')
print("Model training completed and saved as 'student_risk_model.pkl'.")

# Test the model on the testing set
y_pred = stacking_model.predict(x_test)

# Evaluate the model's performance on the test set
perform(y_pred)