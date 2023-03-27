# necessary libraries for the framework
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn import preprocessing
from scipy.stats import bernoulli
from statistics import mean, stdev

class NBA_data:
    '''
    Preprocess the data and prepare it for the ML workflow usage.
    '''
    def __init__(self, year=None):
        '''
        Create data set instance by calling the call the ETL method to load, 
        convert and store the data from the sources in a standard form.
        '''
        self.year=year
        self.ETL()

    def ETL(self):
        '''
        Extract, Transform and Load method.
        This method prepares the data after loading it from the source files 
        and converting it to a standard form which can be used for further analysis.
        '''
        self.load_data()
        self.transform()

    def load_data(self, year=None):
        '''
        Read college player statistics from 2009 to 2022.
        The data can be found in two different csv files, one contains stats from 2009 to 2021
        while the other one contains the latest statistics (2022).

        Read the other data source that contains draft picks from the nba draft 
        for each year from 2009 to 2021.

        After reading all sources, this method also merges them into one data set 
        and creates a df (pandas DataFrame) attribute for the class instance.
        '''
        college2009_2021 = pd.read_csv(
            'Data\CollegeBasketballPlayers2009-2021.csv', low_memory=False)

        college2022 = pd.read_csv(
            'Data\CollegeBasketballPlayers2022.csv', low_memory=False)

        draft = pd.read_excel('Data\DraftedPlayers2009-2021.xlsx')

        # first of all, lets concatenate the college statistical dataframes
        college = pd.concat([college2009_2021, college2022])

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

        # year selection
        if year is not None: 
            self.year = year

    def transform(self):
        '''
        Clean and transform the df attribute of the class instance.
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

        # reorder columns to have drafted_flag as the last column of the dataframe
        col_list = self.df.columns.tolist()
        col_list.pop(-6)  # 'drafted_flag'
        col_list.append('drafted_flag')
        self.df = self.df[col_list]

        # leave only numeric data and fill all remaining columns with zeros
        self.df = self.df.select_dtypes(exclude='object')
        self.df.fillna(value=0, inplace=True)

        # separate predictors from the target feature
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.drafted_flag

        # apply feature scaling for input features using MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        self.X = scaler.fit_transform(self.X)

class random_model:
        '''
        Random model class implementation.
        '''
        def __init__(self,X=NBA_data().X, y=NBA_data().y):
            '''
            Random model instance.

            ## Parameters:
            - data -> for future prediction, 
            the input data's X feature space can be given to the random model manually instead of the X test set.
            '''
            # empirical distribution of the target feature
            self.X = pd.DataFrame(X)
            self.y = y
            self.record_count = self.X.count()
            self.target = self.y[y==1].count()
            self.bernoulli_dist = bernoulli(self.y).rvs(len(self.X))

            # random model with Bernoulli distribution
            self.y_pred = pd.DataFrame(data=[self.bernoulli_dist])
            self.y_pred = self.y_pred.transpose()
            self.y_pred = pd.DataFrame(data=self.y_pred)
            self.y_pred.columns = ['y_pred']
            self.y_pred = np.array(self.y_pred)

        def fit(self,X=None,y=None):
            pass

        def predict(self,X=None):
            return self.y_pred


class logistic_regression:
    '''
    Logistic Regression model class implementation using sklearn's LogisticRegression library.
    '''
    def __init__(self,penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0,
                 warm_start=False, n_jobs=None, l1_ratio=None):
        
        self.model = LR(
            penalty=penalty, 
            dual=dual, 
            tol=tol, 
            C=C, 
            fit_intercept=fit_intercept, 
            intercept_scaling=intercept_scaling, 
            class_weight=class_weight, 
            random_state=random_state, 
            solver=solver, 
            max_iter=max_iter, 
            multi_class=multi_class,
            verbose=verbose, 
            warm_start=warm_start, 
            n_jobs=n_jobs, 
            l1_ratio=l1_ratio
            )

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        self.model.predict(X)

class decision_tree:
    '''
    Decision Tree model class implementation using sklearn's DecisionTreeClassifier library.
    '''
    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
                 max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0):
        
        self.model = DecisionTreeClassifier(
            criterion=criterion, 
            splitter=splitter, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=min_weight_fraction_leaf, 
            max_features=max_features, 
            random_state=random_state, 
            max_leaf_nodes=max_leaf_nodes, 
            min_impurity_decrease=min_impurity_decrease, 
            class_weight=class_weight, 
            ccp_alpha=ccp_alpha
            )

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        self.model.predict(X)

class random_forest:
    '''
    Random Forest model class implementation using sklearn's RandomForestClassifier library.
    '''
    def __init__(self, n_estimators=100, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, bootstrap=True, oob_score=False, max_features='sqrt', 
                 random_state=None, verbose=0, warm_start=False, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, 
                 ccp_alpha=0.0, max_samples=None):
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=min_weight_fraction_leaf, 
            max_features=max_features, 
            random_state=random_state, 
            max_leaf_nodes=max_leaf_nodes, 
            min_impurity_decrease=min_impurity_decrease, 
            bootstrap=bootstrap,
            oob_score=oob_score,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight, 
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
            )

    def fit(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        self.model.predict(X)

class evaluation:
    '''
    Class of evaluation functions.
    '''
    def skf_eval_precision(self, X=NBA_data().X, y=NBA_data().y, model=random_model(), number_of_splits=10, random_state=1):
        '''
        Stratified K-Fold Cross-Validitation evaluation focusing on the precision values.
        '''
        # for the random model it is necessary to ignore cases where the model was unable to predict the target feature
        import warnings
        warnings.filterwarnings('ignore')

        skf = StratifiedKFold(n_splits=number_of_splits, shuffle=True, random_state=random_state)
        eval_score = []

        for train_index, test_index in skf.split(X, y):
            # split X and y
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # fit the model
            model.fit(X_train,y_train)

            # prediction
            y_pred = model.predict()

            # evaluate the model's performance
            # cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
            cr = pd.DataFrame(classification_report(y_pred=y_pred, y_true=y_test, output_dict=True))

            eval_score.append(cr.iloc[1,1]) # precision for drafted_flag = 1 predictions

        return(eval_score)

    def skf_eval_recall(self, X=[], y=[], model=random_model(), number_of_splits=10, random_state=1):
        '''
        Stratified K-Fold Cross-Validitation evaluation focusing on the precision values.
        '''
        # for the random model it is necessary to ignore cases where the model was unable to predict the target feature
        import warnings
        warnings.filterwarnings('ignore')

        skf = StratifiedKFold(n_splits=number_of_splits, shuffle=True, random_state=random_state)
        eval_score = []

        for train_index, test_index in skf.split(X, y):
            # split X and y
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # fit the model
            model.fit(X_train,y_train)

            # prediction
            y_pred = model.predict()

            # evaluate the model's performance
            cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
            cr = pd.DataFrame(classification_report(y_pred=y_pred, y_true=y_test, output_dict=True))

            eval_score.append(cr.iloc[1,1]) # precision for drafted_flag = 1 predictions

        return(eval_score)