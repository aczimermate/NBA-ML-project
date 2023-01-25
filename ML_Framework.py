# necessary libraries for the evaluation framework
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.stats import bernoulli


class Model:
    '''
    This class contains all attributes and methods for classification model construction on the college basketball and NBA draft data sets, and the evaluation of the model instance. 
    As default, it uses a random model based on the target feature's distribution to predict the outcomes. 
    The training of the classification model on the training set followed by the prediction of the y vector based wether on the test feature space or the manually given input data set. 
    ## Parameters:
     - user_defined_model -> the framework handles the following values for this input parameter:
        1. None -> if there is no input value then the framework automatically choses the random model for the classification task
        2. 'lr' -> if the input value is 'lr', then the framework classifies the data by using logistic regression 
    ## Methods:
     - transform: cleaning and transformation of the data set.
     - random_model: construction method of a random model based on the empirical distribution of the target feature vector.
     - split_data: train/test split of the data set.
     - fit: fit method of the classification model on the training data set.
     - fit: fit method of the classification model on the training data set.
     - evaluate: shows the results of the prediction on the test data set/input value(s) compared to the baseline model outcome.
     - run_framework: then runs all methods mentioned above
    '''

    def __init__(self, user_defined_model=None, year=None):

        # read college player statistics from 2009 to 2022
        # the data can be found in two different csv files, one contains stats from 2009 to 2021
        # while the other one contains the latest statistics (2022)
        college1 = pd.read_csv(
            'Data\CollegeBasketballPlayers2009-2021.csv', low_memory=False)
        college2 = pd.read_csv(
            'Data\CollegeBasketballPlayers2022.csv', low_memory=False)

        # the other data source contains draft picks at the nba draft for each year from 2009 to 2021
        draft = pd.read_excel('Data\DraftedPlayers2009-2021.xlsx')

        # first of all, lets concatenate the college statistical dataframes
        college = pd.concat([college1, college2])

        # since the draft data set has merged cells in the table header the first row must be dropped
        draft.drop(0, axis=0, inplace=True)

        # rename the ROUND.1 column to PICK, and modify the PLAYER to player_name
        # so it can be act as a key during the join with the college data set
        draft.rename(
            columns={
                "PLAYER": "player_name",
                "TEAM": "drafted_by",
                "YEAR": "year",
                "ROUND": "draft_round",
                "ROUND.1": "draft_pick"},
            inplace=True)

        # also lower all column names
        draft.columns = draft.columns.str.lower()

        # join (merge) the college set with the draft data to identify those players who have been drafted after playing in college
        self.df = pd.merge(college, draft, how='left',
                           on=['player_name', 'year'])

        # model selection
        if user_defined_model is None:
            self.user_defined_model = None
        elif user_defined_model == 'lr':
            self.user_defined_model = LR()
        # elif user_defined_model == 'dt':
        #     self.user_defined_model = 'decision_tree'

        # year selection
        self.year = year

    def transform(self):
        '''
        Clean and transform the data set.
        '''
        # create a new column to identify the drafted players
        self.df['drafted_flag'] = (~self.df.overall.isnull())*1

        # since the draft data does not contain information about 2022
        # rows for 2022 are removed from the 'df' data set and saved into a new dataframe
        # self.df_2022 = self.df[self.df.year == 2022]

        # during model creation, it is also possible to chose a specific year to filter the data set and use only yearly data
        if self.year is None:
            self.df = self.df[self.df.year < 2022]
        else:
            self.df = self.df[self.df.year == self.year]

        # rename unnamed column 64 for clearity
        self.df.rename(
            columns={'Unnamed: 64': 'player_position'}, inplace=True
        )

        # drop unknown, irrelevant (not statistical, such as 'num': jersey number column) or dupplicate columns
        # unknown with nan values
        self.df = self.df.drop('Unnamed: 65', axis=1)
        # irrelevant, it can be used for more sophisticated prediction tasks
        self.df = self.df.drop('pick', axis=1)
        # irrelevant, it can be used for more sophisticated prediction tasks
        self.df = self.df.drop('overall', axis=1)
        # irrelevant, it can be used for more sophisticated prediction tasks
        self.df = self.df.drop('affiliation', axis=1)
        # irrelevant, it can be used for more sophisticated prediction tasks
        self.df = self.df.drop('draft_round', axis=1)
        # irrelevant, it can be used for more sophisticated prediction tasks
        self.df = self.df.drop('draft_pick', axis=1)
        # irrelevant, not statistical data (jersey number)
        self.df = self.df.drop('num', axis=1)
        # irrelevant, not statistical data (player id in the database)
        self.df = self.df.drop('pid', axis=1)
        # irrelevant, not statistical data (unique value for all rows)
        self.df = self.df.drop('type', axis=1)

        # handle mistyped or wrong values
        self.df.yr.replace('0', 'None', inplace=True)
        self.df.yr.replace('57.1', 'None', inplace=True)
        self.df.yr.replace('42.9', 'None', inplace=True)

        # handle missing values
        self.df.drafted_flag.fillna(value=0, inplace=True)
        self.df.yr.fillna(value='None', inplace=True)
        self.df.player_position.fillna(value='None', inplace=True)

        # one hot encode categorical column: yr
        self.df = pd.get_dummies(self.df, columns=['yr'])

        # leave only numeric data and fill all remaining columns with zeros
        self.df = self.df.select_dtypes(exclude='object')
        self.df.fillna(value=0, inplace=True)

    def random_model(self, input_value=None):
        '''
        Random model implementation.
        ## Parameters:
         - input_value -> for future prediction, the input X feature space can be given to the random model manually instead of the X test set
        '''
        # empirical distribution of the target feature
        self.drafted_per_year = self.df[self.df.drafted_flag == 1].groupby('year').count()[
            'drafted_flag']
        self.count_per_year = self.df.groupby('year').count()['GP']
        self.yearly_drafts = self.drafted_per_year/self.count_per_year

        # probability of drafted flag = 1
        self.prob_drafted = np.average(self.yearly_drafts)

        # random model with the given Bernoulli distribution
        if input_value is None:
            self.rand_model = pd.DataFrame(
                data=[bernoulli(self.prob_drafted).rvs(len(self.X_test))])
        else:
            self.rand_model = pd.DataFrame(
                data=[bernoulli(self.prob_drafted).rvs(len(input_value))])

        self.rand_model = self.rand_model.transpose()
        self.rand_model = pd.DataFrame(data=self.rand_model)
        self.rand_model.columns = ['pred_drafted_flag']

        return np.array(self.rand_model)

    def split_data(self, test_size=.2):
        '''
        Split the data into training a test sets, then return the splitted data set.
        ## Parameters:
         - test_size -> train/test ratio [0,1]; default = 0.2
        '''
        # separate predictors from the target feature
        X = self.df.iloc[:, :-1]
        y = self.df.drafted_flag

        # Split to training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size)

    def fit(self):
        '''
        Fit the chosen classification model to the training data.
        '''
        if self.user_defined_model is None:
            pass
        else:
            self.model = self.user_defined_model.fit(
                self.X_train, self.y_train)

    def predict(self, input_value=None):
        '''
        Predict the target feature: either manual input or test data.
        ## Parameters:
         - input_value -> for future prediction, the input X feature space can be given to the random model manually instead of the X test set
        ## Output:
         - result -> prediction vector of the target feature
        '''
        # determine the predictor set
        if input_value is None:
            predictors = self.X_test
        else:
            predictors = np.array([input_value])

        # predict the outcome with the selected model
        if self.user_defined_model is None:
            result = self.random_model(input_value=predictors)
        else:
            result = self.user_defined_model.predict(predictors)

        return result

    def evaluate(self, y_pred):
        '''
        Calculate the prediction results of the classification model.
        ## Parameters:
         - y_pred -> for evaluation between real and predicted values, it is necessary to provide the method the predicited target vector
        ## Output:
         - cm -> confusion matrix
         - cm_display -> visual representation of the confusion matrix
         - cp -> classification report
        '''
        # calculate the confusion matrix and create classification report
        conf_matrix = confusion_matrix(y_pred=y_pred, y_true=self.y_test)
        report = classification_report(
            y_pred=y_pred, y_true=self.y_test, output_dict=True
        )
        class_report = pd.DataFrame(report)
        # cm_display = ConfusionMatrixDisplay(
        #     confusion_matrix=confusion_matrix, display_labels=[False, True])

        return conf_matrix, class_report

    def run_framework(self, input_value=None, test_size=.2):
        '''
        Run all methods in the right order.
        ## Parameters:
         - input_value -> for future prediction, the input X feature space can be given to the random model manually instead of the X test set
         - test_size -> train/test ratio [0,1]; default = 0.2
        ## Output:
         - cm -> confusion matrix
         - cm_display -> visual representation of the confusion matrix
         - cp -> classification report
         - result -> prediction vector of the target feature
        '''
        # transform and split the data set
        self.transform()
        self.split_data(test_size=test_size)

        # check the existance of the input vector
        if input_value is None:
            predictors = self.X_test
        else:
            predictors = np.array([input_value])

        # construct random model for default usage
        self.random_model(input_value=predictors)

        # fit the model on the training data
        self.fit()

        # predict the target vector values and evaluate the prediction
        if self.user_defined_model is None:
            result = self.random_model(input_value=predictors)
        else:
            result = self.user_defined_model.predict(predictors)

        cm, cp = self.evaluate(result)

        return result, cm, cp
