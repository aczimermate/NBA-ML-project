# necessary libraries for the evaluation framework
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, classification_report

class LogRegModel:
    '''
    Creating a logistic regression class instance indicates a train/test split of the data 
    and training the classification model on the training set followed by a prediction on the test values.

    The class uses methods from the sklearn libraries below: 
     - sklearn.linear_model.LogisticRegression
     - sklearn.model_selection
     - sklearn.metrics.confusion_matrix
     - sklearn.metrics.classification_report


    Parameters:
     - Data set containing the X feature space predictors and the y answer vector. 
       The train/test split takes place during the evaluation process.
       
    Methods:
     - X_train:
     - X_test:
     - y_train:
     - y_test:
     - model:
     - test_results:
    '''

    def __init__(self, X, y, test_size):
        
        # split the data into training a test sets
        X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=test_size)

        # create instance features for later usage
        self.X_train = np.to_array(X_train)
        self.X_test = np.to_array(X_test)
        self.y_train = np.to_array(y_train)
        self.y_test = np.to_array(y_test)

        # fit the logistic regression model to the training data
        self.model = LR().fit(X_train,y_train)
        self.prediction = self.model.predict(X_test)
        self.prediction_results = self.prediction.score()


    def X_train(self):
        return (self.X_train)

    def X_test(self):
            return (self.X_test)

    def y_train(self):
            return (self.y_train)

    def y_test(self):
            return (self.y_test)

    def model_params(self):
        '''
        Return the training model parameters. 
        '''
        return self.model.get_params()

    def results(self):
        '''
        Return the test model evaluation results. 
        '''
        score = self.prediction_results
        confusion_matrix
        classification_report
        return score

    class ClassEvaluator:
        '''
        '''
        def __init__(self,model):
             self.model = model
