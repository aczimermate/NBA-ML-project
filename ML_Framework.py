# necessary libraries for the framework
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn import preprocessing
from scipy.stats import bernoulli
from statistics import mean, stdev

# Classes of machine learning models and data loading / preprocessing step applications.

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

class NBA_data_ext:
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
                "ROUND" : "draft_round", 
                "ROUND.1" : "pick_no",
                "OVERALL" : "draft_pick"},
            inplace=True)

        # also lower all column names
        draft.columns = draft.columns.str.lower()

        # join (merge) the college set with the draft data to identify those players who have been drafted after playing in college
        self.df = pd.merge(college, draft, how='outer',
                           on=['player_name', 'year'])

        # year selection
        if year is not None: 
            self.year = year

    def transform(self):
        '''
        Clean and transform the df attribute of the class instance.
        '''
        # create a new column to identify the drafted players
        self.df['drafted_flag'] = (~self.df.draft_pick.isnull())*1

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
        self.df = self.df.drop('Unnamed: 65', axis=1)
        # irrelevant, not statistical data (jersey number)
        self.df = self.df.drop('num', axis=1)
        # irrelevant, not statistical data (player id in the database)
        self.df = self.df.drop('pid', axis=1)
        # irrelevant, not statistical data (unique value for all rows)
        self.df = self.df.drop('type', axis=1)
        # irrelevant, the information is included in the draft_pick column
        self.df = self.df.drop('pick_no', axis=1)
        # irrelevant, the information is included in the draft_pick column
        self.df = self.df.drop('pick', axis=1)

        # handle mistyped or wrong values
        self.df.yr.replace('0', 'None', inplace=True)
        self.df.yr.replace('57.1', 'None', inplace=True)
        self.df.yr.replace('42.9', 'None', inplace=True)

        # handle missing values
        self.df.drafted_flag.fillna(value=0, inplace=True)
        self.df.yr.fillna(value='None', inplace=True)
        self.df.player_position.fillna(value='None', inplace=True)
        self.df.draft_pick.fillna(value=0, inplace=True)
        self.df.draft_round.fillna(value=0, inplace=True)

        # one hot encode categorical column: yr
        self.df = pd.get_dummies(self.df, columns=['yr'])

        # reorder columns to have drafted_flag as the last column of the dataframe
        col_list = self.df.columns.tolist()
        col_list.pop(-7)  # 'draft_pick'
        col_list.append('draft_pick')
        self.df = self.df[col_list]

        # leave only numeric data and fill all remaining columns with zeros
        self.df = self.df.select_dtypes(exclude='object')
        self.df.fillna(value=0, inplace=True)

        # separate predictors from the target feature
        self.X = self.df.iloc[:, :-1]
        self.y = np.array(self.df.iloc[:, -1])

        # apply feature scaling for input features using MinMaxScaler
        scaler = preprocessing.StandardScaler()
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

            # in case our goal is to predict drafted players:
            if max(self.y) == 1:
                self.target = self.y[y==1].count()
                self.bernoulli_dist = bernoulli(self.y).rvs(len(self.X))
                 
                 # random model with Bernoulli distribution
                self.y_pred = pd.DataFrame(data=[self.bernoulli_dist])
                self.y_pred = self.y_pred.transpose()
                self.y_pred = pd.DataFrame(data=self.y_pred)
                self.y_pred.columns = ['y_pred']
                self.y_pred = np.array(self.y_pred)

            # in case our goal is to predict pick numbers:
            else:
                self.target = self.y
                self.y_pred = np.random.randint(0,61,len(self.y))

        def fit(self,X=None,y=None):
            return self

        def predict(self,X=None):
            return self.y_pred

class logistic_regression:
    '''
    Logistic Regression model class implementation using sklearn's LogisticRegression library.
    '''
    def __init__(self, penalty=['l2'], dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver=['lbfgs'], max_iter=100, multi_class=['auto'], verbose=0,
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
        return self

    def predict(self,X):
        return self.model.predict(X)

class decision_tree:
    '''
    Decision Tree model class implementation using sklearn's DecisionTreeClassifier library.
    '''
    def __init__(self, criterion=['gini'], splitter=['best'], max_depth=None, min_samples_split=2, 
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
        return self

    def predict(self,X):
        return self.model.predict(X)

class random_forest:
    '''
    Random Forest model class implementation using sklearn's RandomForestClassifier library.
    '''
    def __init__(self, n_estimators=100, criterion=['gini'], splitter='best', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, bootstrap=True, oob_score=False, max_features=['sqrt'], 
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
        return self

    def predict(self,X):
        return self.model.predict(X)

class xgboost:
    '''
    XGBoost model class implementation using xgboost's XGBClassifier library.
    '''
    def __init__(self, base_score=0.5, colsample_bylevel=1, colsample_bytree=1, 
                 gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
                 min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
                 objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, seed=0, silent=True, subsample=1):

        self.model = XGBClassifier(
            base_score=base_score,
            colsample_bylevel=colsample_bylevel, 
            colsample_bytree=colsample_bytree, 
            gamma=gamma, 
            learning_rate=learning_rate, 
            max_delta_step=max_delta_step, 
            max_depth=max_depth, 
            min_child_weight=min_child_weight, 
            missing=missing, 
            n_estimators=n_estimators, 
            nthread=nthread,
            objective=objective,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight, 
            seed=seed,
            silent=silent,
            subsample=subsample
            )

    def fit(self,X,y):
        self.model.fit(X,y)
        return self

    def predict(self,X):
        return self.model.predict(X)

# Functions implemented for evaluation, visualization and hyperparameter tuning.

def warnings_off():
    '''
    The following functions calculate evaluation measures for 
    the trained models using Stratified K-Fold Cross-Validation technique.
    '''

    # for the evaluation of the models' performance it is necessary to 
    # ignore cases where the model was unable to predict the target feature at all
    import warnings
    warnings.filterwarnings('ignore')

def skf_cross_val(X=NBA_data().X, y=NBA_data().y, model=random_model, number_of_splits=10, random_state=1):
    '''
    Stratified K-Fold Cross-Validitation evaluation.
    '''
    warnings_off()
    skf = StratifiedKFold(n_splits=number_of_splits, shuffle=True, random_state=random_state)
    conf_matrix = []
    precision = []
    accuracy = []
    recall = []
    f1_score = []
    # class_rep = []

    for train_index, test_index in skf.split(X, y):
        # split X and y
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # fit the model
        model.fit(X=X_train,y=y_train) # type: ignore

        # prediction
        y_pred = model.predict(X=X_test) # type: ignore

        # evaluate the model's performance
        cr = pd.DataFrame(classification_report(y_pred=y_pred, y_true=y_test, output_dict=True))

        # confusion matrices
        conf_matrix.append(confusion_matrix(y_true=y_test,y_pred=y_pred))
        
        # precisions for drafted_flag = 1 predictions
        precision.append(cr.iloc[1,1])

        # accuracies
        accuracy.append(cr.iloc[2,1])

        # recall values
        recall.append(cr.iloc[2,2])
        
        # F1-scores
        f1_score.append(cr.iloc[2,3])

        # class_rep.append(cr)

    return(conf_matrix,precision,accuracy,recall,f1_score,model)

def boxplot_eval_scores(eval_data=[0,0,0,0],fig_height=10, fig_width=10, colors = ['#0194fe', '#d8cabf','#b9d090', '#fb9329']):
    '''
    Create and show boxplot for every evaluation score arrays (precision, accuracy, recall and F1-score) 
    with the given height and width and color palette.
    '''
    # setup the figure
    fig = plt.figure(figsize =(fig_height, fig_width))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(eval_data, patch_artist = True, notch =True, vert = False)

    # colors for the plots
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color) # type: ignore

    # changing color and linewidth of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B', linewidth = 1.5, linestyle =":")

    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)

    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['Precision', 'Accuracy', 'Recall', 'F1-score'])
    # ax.set_xticklabels([np.arange(0, 1, .2)])

    # Add title
    plt.title("Evaluation box plots")

    # Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Add grid to the plot
#    ax.grid(linestyle='-', linewidth=.5)
    
    # Major ticks every 0.20, minor ticks every 5
    major_ticks = np.arange(0, 1.1, .1)
    ax.set_xticks(major_ticks)
#    ax.set_yticks(major_ticks)
#    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both',axis='both')
    
    # show plot
    plt.show()

def gridsearch_cv(X_train=NBA_data().X, y_train=NBA_data().y, X_test=NBA_data().X, y_test=NBA_data().y, model=RandomForestClassifier(), number_of_splits=10, random_state=1, param_grid={}, scoring='balanced_accuracy'):
    '''
    Grid Search Cross-Validitation function.
    '''
    warnings_off()

    conf_matrix = []
    precision = []
    accuracy = []
    recall = []
    f1_score = []
   
    # Grid search cross validation model
    gs_cv = GridSearchCV(
        estimator=model,
        param_grid = param_grid,
        cv=number_of_splits,
        scoring=scoring,
        verbose=2,
        n_jobs=-1
        )
    
    # fit the model
    gs_cv.fit(X_train,y_train)

    # prediction
    y_pred = gs_cv.predict(X_test)

    # confusion matrix
    conf_matrix.append(confusion_matrix(y_true=y_test,y_pred=y_pred))

    # evaluate the model's performance
    cr = pd.DataFrame(classification_report(y_pred=y_pred, y_true=y_test, output_dict=True))

    # precisions for drafted_flag = 1 predictions
    precision.append(cr.iloc[1,1])

    # accuracies
    accuracy.append(accuracy_score(y_pred=y_pred, y_true=y_test))

    # recall values
    recall.append(cr.iloc[2,2])
    
    # F1-scores
    f1_score.append(cr.iloc[2,3])

    return(conf_matrix,precision,accuracy,recall,f1_score,gs_cv)
