import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn import svm
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv("flask_ml/framingham.csv")

# Drop duplicate values
data = data.drop_duplicates()

# Split data into features and target
X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define feature columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Create a transformer for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer()),  # Handle missing values using KNN
    ('scaler', RobustScaler())  # Apply robust scaling
])

# Create a transformer for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values for categorical features
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One hot encoding
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to the data
X_preprocessed = preprocessor.fit_transform(X)

# Apply feature selection
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_preprocessed, y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
# kernel_inp = 'linear'
# C_inp=1.0
# gamma_inp = 'scale'
# Create and fit the model
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=70, learning_rate = 1)
model.fit(X_train, y_train)
# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
# Compute Mean Squared Error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Scale the MSE if needed (assuming you want a normalized score)
scaling_factor = np.mean((y_train - np.mean(y_train))**2)  # Example scaling factor
smse_train = mse_train / scaling_factor
smse_test = mse_test / scaling_factor

print(f"Training SMSE: {smse_train}")
print(f"Test SMSE: {smse_test}")

print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Save the model to a pickle file
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved to svm_model.pkl")