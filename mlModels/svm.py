from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

def svm(X_train, X_test, y_train, y_test):
    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_rmse = np.sqrt(mean_squared_error(y_test, svm_preds))
    return {'RMSE': svm_rmse}
