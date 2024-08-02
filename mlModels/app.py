# import pandas as pd
# import numpy as np
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import OneHotEncoder, RobustScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2
# from scipy.stats.mstats import winsorize
# import autosklearn.classification
# # Function to handle outliers using winsorization
# df = pd.read_csv('Iris.csv')

# def winsorize_data(df, limits=[0.01, 0.99]):
#     for col in df.select_dtypes(include=[np.number]).columns:
#         df[col] = winsorize(df[col], limits=limits)
#     return df

# # Sample DataFrame
# # df = pd.read_csv('your_dataset.csv')

# # Drop duplicates
# df = df.drop_duplicates()

# # Winsorize outliers
# df = winsorize_data(df)

# # Separate features and target
# X = df.drop(columns=['E'])
# y = df['E']

# # Identify numeric and categorical features
# numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
# categorical_features = X.select_dtypes(include=[object]).columns.tolist()

# # Preprocessing pipeline
# numeric_transformer = Pipeline(steps=[
#     ('imputer', KNNImputer(n_neighbors=3)),
#     ('scaler', RobustScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Apply preprocessing
# X_processed = preprocessor.fit_transform(X)

# # Convert the processed features back to a DataFrame
# X_processed = pd.DataFrame(X_processed.toarray(), columns=numeric_features + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_features)))

# # Feature selection using Chi-Square
# selector = SelectKBest(chi2, k='all')
# X_selected = selector.fit_transform(X_processed, y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# # Use Auto-sklearn for AutoML
# automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=300, random_state=42)
# automl.fit(X_train, y_train)

# print("Best models found by auto-sklearn:")
# print(automl.leaderboard())

# print("Test set score: ", automl.score(X_test, y_test))


# import pandas as pd
# import numpy as np
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import VarianceThreshold
# from scipy.stats.mstats import winsorize
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.metrics import accuracy_score

# from tpot import TPOTClassifier

# df = pd.read_csv("flask_ml/framingham.csv")

# def winsorize_data(df, limits=[0.01, 0.99]):
#     for col in df.select_dtypes(include=[np.number]).columns:
#         df[col] = winsorize(df[col], limits=limits)
#     return df

# # Drop duplicates
# df = df.drop_duplicates()

# # Winsorize outliers
# df = winsorize_data(df)

# # Separate features and target
# X = df.drop(columns=['TenYearCHD'])
# y = df['TenYearCHD']

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Identify numeric and categorical features
# numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
# categorical_features = X.select_dtypes(include=[object]).columns.tolist()

# # Preprocessing pipeline
# numeric_transformer = Pipeline(steps=[
#     ('imputer', KNNImputer(n_neighbors=3)),
#     ('scaler', RobustScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Fit the preprocessor and transform the data
# X_processed = preprocessor.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)
# print("Classes in y_train:", np.unique(y_train))

# config_dict = {
#     'sklearn.svm.SVC': {
#         'C': [1.0, 10.0, 100.0],
#         'kernel': ['linear', 'rbf'],
#         'gamma': ['scale', 'auto']
#     }
# }

# # Initialize TPOT with the custom configuration
# tpot = TPOTClassifier(
#     config_dict=config_dict,
#     verbosity=2,
#     generations=5,
#     population_size=20,
#     random_state=42,
#     # Using a smaller max time to limit execution time
#     max_time_mins=10
# )

# # Fit the TPOT model
# tpot.fit(X_train, y_train)

# # Evaluate the model
# y_pred = tpot.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")

# # Show the final pipeline found by TPOT
# # print(tpot.fitted_pipeline_)

# # Export the pipeline
# tpot.export('tpot_pipeline.py')

# import sys
# import json

# import numpy as np
# import pandas as pd
# from sklearn.impute import KNNImputer, SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pickle
# import math
# from sklearn import svm
# from sklearn.metrics import mean_squared_error

# def main():
#     csv_file_path = sys.argv[1]
    
#     # Read the CSV file using pandas
#     data = pd.read_csv(csv_file_path)
    
#     json_string = sys.argv[1]
#     data = json.loads(json_string)
#     variable1 = data.get('variable1')
#     variable2 = data.get('variable2')
#     variable3 = data.get('variable3')
    
#     preprocessing(data)
    
#     X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
#     if model_name == "random_forest":
#         from sklearn.ensemble import RandomForestClassifier
#         model = RandomForestClassifier(n_estimators=100) #max_depth=20, max_features='log2', min_samples_split=2, min_samples_leaf=2)
#         model.fit(X_train, y_train)

#     elif model_name == "XGBoost":
#         import xgboost as xgb
#         from xgboost import XGBClassifier, XGBRegressor

#         model = XGBClassifier(
#             objective='binary:logistic',  # for binary classification
#             max_depth=5,  # maximum depth of trees
#             learning_rate=0.1,  # learning rate
#             n_estimators=100,  # number of boosting rounds
#             eval_metric='logloss'  # evaluation metric
#         )
#         model.fit(X_train, y_train)

#     elif model_name == "AdaBoost":
#         from sklearn.ensemble import AdaBoostClassifier
#         model = AdaBoostClassifier(n_estimators=70, learning_rate = 1)
#         model.fit(X_train, y_train)

#     elif model_name == "svm":
#         model = svm.SVC(kernel='linear', C=1.0, gamma='scale')
#         model.fit(X_train, y_train)
        
#     elif model_name == "decision_tree":
#         from sklearn.tree import DecisionTreeClassifier
#         model = DecisionTreeClassifier()
#         model.fit(X_train, y_train)


    
#     result = {"message": "Hello, " + data["name"], "age": data["age"]}


# def preprocessing(data):
#     # Drop duplicate values
#     data = data.drop_duplicates()

#     # Split data into features and target
#     X = data.drop('TenYearCHD', axis=1)
#     y = data['TenYearCHD']

#     # Encode the target variable
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y)

#     # Define feature columns
#     categorical_features = X.select_dtypes(include=['object']).columns
#     numerical_features = X.select_dtypes(exclude=['object']).columns

#     # Create a transformer for numerical features
#     numerical_transformer = Pipeline(steps=[
#         ('imputer', KNNImputer()),  # Handle missing values using KNN
#         ('scaler', RobustScaler())  # Apply robust scaling
#     ])

#     # Create a transformer for categorical features
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values for categorical features
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One hot encoding
#     ])

#     # Combine transformers into a preprocessor
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer, numerical_features),
#             ('cat', categorical_transformer, categorical_features)
#         ])

#     # Apply preprocessing to the data
#     X_preprocessed = preprocessor.fit_transform(X)

#     # Apply feature selection
#     selector = SelectKBest(score_func=f_classif, k='all')
#     X_selected = selector.fit_transform(X_preprocessed, y)

#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# def evaluate():
#     # Make predictions and evaluate the model
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Model accuracy: {accuracy}")

#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)


#     train_accuracy = accuracy_score(y_train, y_train_pred)
#     test_accuracy = accuracy_score(y_test, y_test_pred)
#     # Compute Mean Squared Error
#     rmse_train = math.sqrt(mse_train)
#     rmse_test = math.sqrt(mse_test)

#     print(f"Training RMSE: {rmse_train}")
#     print(f"Test RMSE: {rmse_test}")

#     print(f"Training accuracy: {train_accuracy}")
#     print(f"Test accuracy: {test_accuracy}")

#     # Save the model to a pickle file
#     with open('svm_model.pkl', 'wb') as file:
#         pickle.dump(model, file)

#     print("Model saved to svm_model.pkl")


import sys
import json
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import math
from sklearn import svm

def load_data(csv_file_path, json_string):
    # Read the CSV file using pandas
    data_csv = pd.read_csv(csv_file_path)
    
    # Parse JSON string and create a DataFrame
    data_json = json.loads(json_string)
    data_json_df = pd.DataFrame([data_json])
    
    # Combine both CSV and JSON data
    data = pd.concat([data_csv, data_json_df], ignore_index=True)
    return data

def preprocessing(data):
    # Drop duplicate values
    data = data.drop_duplicates()

    # Split data into features and target
    X = data.drop('target', axis=1)
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

    return X_selected, y


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute Mean Squared Error
    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, y_pred)
    
    rmse_train = math.sqrt(mse_train)
    rmse_test = math.sqrt(mse_test)

    print(f"Model accuracy: {accuracy}")
    print(f"Training RMSE: {rmse_train}")
    print(f"Test RMSE: {rmse_test}")

    print(f"Training accuracy: {accuracy_score(y_train, model.predict(X_train))}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred)}")

    # Save the model to a pickle file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model saved to model.pkl")
def main():
    action = sys.argv[1]
    csv_file_path = sys.argv[2]
    if action == 'show-data-dimensions':
        data = load_data(csv_file_path)
        dimensions = {
            "numRows": data.shape[0],
            "numColumns": data.shape[1]
        }
        print(json.dumps(dimensions))
    elif action == 'remove-features':
        features_to_remove = sys.argv[3]
        # Implement the logic to remove features
    elif action == 'convert-numbers':
        features_to_convert = sys.argv[3]
        # Implement the logic to convert features to numbers
    elif action == 'train-test-split':
        train_data_percentage = float(sys.argv[3])
        test_data_percentage = float(sys.argv[4])
        # Implement the logic to perform train-test split
    elif action == 'train-random-forest':
        criterion = sys.argv[3]
        max_depth = int(sys.argv[4])
        n_estimators = int(sys.argv[5])
        data = load_data(csv_file_path)
        X_selected, y = preprocessing(data)
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
        model.fit(X_train, y_train)
        evaluate_model(model, X_train, y_train, X_test, y_test)
    

    # Add other model training actions here

if __name__ == '__main__':
    main()


# def main():
#     csv_file_path = sys.argv[1]
    
#     # Read the CSV file using pandas
#     data = pd.read_csv(csv_file_path)
    
#     json_string = sys.argv[2]
#     parameters = json.loads(json_string)
#     model_name = parameters.get('variable1')


#     X_selected, y, preprocessor = preprocessing(data)
    
#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
#     # Initialize and train the model based on the model_name
#     if model_name == "random_forest":
#         n_est = parameters.get('variable2')
#         depth = parameters.get('variable3')

#         from sklearn.ensemble import RandomForestClassifier
#         model = RandomForestClassifier(n_estimators=100)
#     elif model_name == "XGBoost":
#         import xgboost as xgb
#         from xgboost import XGBClassifier
#         model = XGBClassifier(
#             objective='binary:logistic',
#             max_depth=5,
#             learning_rate=0.1,
#             n_estimators=100,
#             eval_metric='logloss'
#         )
#     elif model_name == "AdaBoost":
#         from sklearn.ensemble import AdaBoostClassifier
#         model = AdaBoostClassifier(n_estimators=70, learning_rate=1)
#     elif model_name == "svm":
#         model = svm.SVC(kernel='linear', C=1.0, gamma='scale')
#     elif model_name == "decision_tree":
#         from sklearn.tree import DecisionTreeClassifier
#         model = DecisionTreeClassifier()
#     else:
#         raise ValueError("Invalid model name")
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Evaluate the model
#     evaluate_model(model, X_train, y_train, X_test, y_test)

