# necessary libraries for the evaluation framework
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, classification_report

class evaluation_framework:
    '''
    This class contains all attributes and necessary procedures for classification model construction and evaluation. 
    The training of classification model on the training set followed by the prediction of the y vector based on the test feature space.

    The class uses methods from the scikit-learn libraries listed below: 
     - sklearn.linear_model.LogisticRegression
     - sklearn.model_selection
     - sklearn.metrics.confusion_matrix
     - sklearn.metrics.classification_report

    Input parameters:
     - Data set containing the X feature space predictors and the y answer vector, and the desired test size for the train/test split. 
       
    Methods:
     - split_data: train/test split of the data set
     - classify: construction method of the classification model
     - evaluate: evaluation of the test results 
    '''

    def split_data(X, y, test_size):
        '''
        Split the data into training a test sets, then return the splitted data set.
        '''
        X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=test_size)
        return X_train, y_train, X_test, y_test

    def classify(X_train, y_train, X_test, y_test):
        '''
        Fit the classification model to the training data.
        '''
        model = LR().fit(X_train,y_train)
        y_pred = model.predict(X_test)
        return y_pred

    def evaluate(y_pred, y_test):
        '''
        Return the evaluation results of the classification model. 
        '''
        cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
        cp = classification_report(y_pred=y_pred, y_true=y_test)
        return cm,cp
