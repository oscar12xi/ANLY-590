# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:22:31 2019

@author: Jimny
"""

## Import needed packages
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import linear_model
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV   #Perforing grid search

from sklearn.model_selection import train_test_split

from operator import itemgetter 

import pickle

from sklearn.metrics import mutual_info_score,roc_auc_score,confusion_matrix,accuracy_score, roc_curve, auc, classification_report

import itertools
from scipy import interp

import xgboost as xgb
import lightgbm as lgb

from mlens.ensemble import SuperLearner
from sklearn.ensemble import RandomForestClassifier

# Define the main function
def main():
    # Open and read in train x, train y, and scaled test data
    with open('AviationData_cleaned_V3.csv', 'r') as input_all:
        df_raw = pd.read_csv(input_all, encoding = 'utf-8')
    
    # Final check on NA values from 
    print('Check number of NA values from selected columns:\n',
          df_raw.isnull().sum())
    
    # Drop rows containing NA values and reset index
    df_raw.dropna(axis=0, inplace = True)
    df_raw.reset_index(drop = True, inplace = True)
    
    # Prepare response label
    df_raw['Injury Severity']= df_raw['Injury Severity'].replace('Incident', 'Non-Fatal') 

    # Separate the two classes in the original dataset
    df_none = df_raw.loc[df_raw['Injury Severity'] == 'Non-Fatal']
    df_fatl = df_raw.loc[df_raw['Injury Severity'] == 'Fatal']
    
    # Balance Dataset
    n_fatl = len(df_fatl)
    df_none = df_none.sample(n = n_fatl, replace = False, random_state = 117)
    
    # Re-construct dataset
    df_sampled = pd.concat([df_none,df_fatl], ignore_index=True)
    df_sampled.reset_index(drop = True, inplace = True)

    # Separate predictors and response
    df_X = df_sampled.drop(['Injury Severity', 'Airport Code'], axis = 1)
    df_y = df_sampled.loc[: ,  'Injury Severity' ]
    
    # Convert string response to numerical response fro convenience
    df_y.replace('Non-Fatal', '0', inplace = True)
    df_y.replace('Fatal', '1', inplace = True)
    
    # Define and apply one-hot encoder to encode predictors
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_X)
    df_X = pd.DataFrame(enc.transform(df_X).toarray(), columns = enc.get_feature_names(list(df_X.columns)))
    
    # Separate train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.5, random_state=1378)
    
    # Recude dataset dimension
    #X_train, X_test = dimension_reduction(X_train, y_train, X_test, 80 , method = 'PCA')
  
    # Define MLP classifier
    clf_mlp = MLPClassifier(hidden_layer_sizes=(100), activation='relu', solver='adam', 
                            alpha=0.0001, batch_size='auto', learning_rate='constant', 
                            learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
                            random_state=117, tol=0.0001, verbose=False, warm_start=False, 
                            momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                            n_iter_no_change=10)
    
    # Define XGBoost classifier
    clf_xgb = xgb.XGBClassifier(booster='gbtree',
                               objective= 'binary:logistic',
                               eval_metric='logloss',
                               tree_method= 'auto',
                               max_depth= 6,
                               min_child_weight= 1,
                               gamma = 0,
                               subsample= 1,
                               colsample_bytree = 1,
                               reg_alpha = 0,
                               reg_lambda = 1,
                               learning_rate = 0.1,
                               seed=27)
    
    # Define LGB Classifier
    clf_lgb = lgb.LGBMClassifier(objective = 'binary',
                                    boosting = 'gbdt',
                                    metric = 'binary_logloss',
                                    num_leaves = 15,
                                    min_data_in_leaf = 10,
                                    max_depth = 5,
                                    bagging_fraction = 0.85,
                                    bagging_freq = 11,
                                    feature_fraction = 0.5,
                                    lambda_l1 = 0.01,
                                    lambda_l2 = 0.3,
                                    num_iterations = 100,
                                    learning_rate = 0.08,
                                    random_state = 117)
    
    # Define random forest classifier
    clf_rf = RandomForestClassifier(n_estimators=300, criterion='gini', 
                                    max_depth=None, min_samples_split=2, 
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                    max_features='auto', random_state = 117)
    
    
    # Fit base learners using whole train dataset
    clf_mlp.fit(X_train,y_train)
    clf_xgb.fit(X_train,y_train)
    clf_lgb.fit(X_train,y_train)
    clf_rf.fit(X_train,y_train)
    
    # Generate predicted probability using base learners
    mlp_proba = clf_mlp.predict_proba(X_test)[:, 1]
    xgb_proba = clf_xgb.predict_proba(X_test)[:, 1]
    lgb_proba = clf_lgb.predict_proba(X_test)[:, 1]
    rf_proba = clf_lgb.predict_proba(X_test)[:, 1]
    
    # Initialize prediction using base learners' results
    pred_mlp = pd.Series(np.full(len(y_test), 0)) 
    pred_xgb = pd.Series(np.full(len(y_test), 0)) 
    pred_lgb = pd.Series(np.full(len(y_test), 0)) 
    pred_rf = pd.Series(np.full(len(y_test), 0)) 
    
    # Set threshold
    thres_mlp = 0.5
    thres_xgb = 0.5
    thres_lgb = 0.5
    thres_rf = 0.5
    
    # Make final prediction
    pred_mlp[mlp_proba >= thres_mlp] = 1
    pred_xgb[xgb_proba >= thres_xgb] = 1
    pred_lgb[lgb_proba >= thres_lgb] = 1
    pred_rf[rf_proba >= thres_rf] = 1
    
    # Map test data response into integers
    y_test = list(map(int, y_test))
    
    # Generate prediction report using base learners
    print('\n\nMLP:')
    print_validate(y_test, pred_mlp)
    
    print('\n\nXGB:')
    print_validate(y_test, pred_xgb)
    
    print('\n\nLGB:')
    print_validate(y_test, pred_lgb)
    
    print('\n\nRF:')
    print_validate(y_test, pred_rf)
    
    # Set base learner dictionary
    base_learners = {'mlp': clf_mlp,
                    'xgb': clf_xgb,
                    'lgb' : clf_lgb,
                    'rf': clf_rf
                    }
    
    # Define super learner
    sup_learner = SuperLearner(
                random_state=117
                )
    
    # Add the base learners and the meta learner
    sup_learner.add(list(base_learners.values()), proba = True)
    sup_learner.add_meta(linear_model.BayesianRidge(alpha_1 = 1e-3))
    
    # Train the ensemble
    sup_learner.fit(X_train,y_train)
    
    # Make prediction using super learner
    sl_proba = sup_learner.predict_proba(X_test)
    pred_sl = pd.Series(np.full(len(y_test), 0)) 
    thres_sl = 0.5
    pred_sl[sl_proba >= thres_sl] = 1
    
    print('\n\nSL:')
    print_validate(y_test, pred_sl)
    
    # ROC Curves for test dataset
    plt.figure(figsize=(8,7))
    draw_roc(y_test, sl_proba, 'Super Learner', 'tab:cyan', '-')
    draw_roc(y_test, mlp_proba, 'MLP NN', 'royalblue', '-')
    draw_roc(y_test, xgb_proba, 'XGBoost', 'lightcoral', '--')
    draw_roc(y_test, lgb_proba, 'LightGBM', 'seagreen', '-.')
    draw_roc(y_test, rf_proba, 'Random Forest', 'darkorange', '-')
    
    plt.plot([0, 1], [0, 1], 'k--', lw = 4)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Test Result')
    plt.legend(loc="lower right", fontsize = 14, handlelength=4)
    plt.show()
    
    
# This function is used to draw ROC curves
def draw_roc(y_true, y_proba, clf_name, col, style):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr,  lw=4, color = col, linestyle = style, alpha = 0.5,
             label='ROC of {0} (AUC = {1:0.2f})'
                 ''.format(clf_name, roc_auc))
    
# This function is used to print validation/test resuls
def print_validate(Y_val, predictions):
    print("\nAccuracy score: ",accuracy_score(Y_val, predictions))
    print("\nConfusion Matrix: \n",confusion_matrix(Y_val, predictions))
    print(classification_report(Y_val, predictions))

# Define the function to reduce data dimensionality
def dimension_reduction(train_x, train_y, test_x, n_col, method = 'fact'):
    # Obtain column names
    attr_list = train_x.columns
    
    # Using RFE to rank feactures and then select
    if method == 'RFE':
        # Using RFE to rank attributes
        lin_reg = LinearRegression()
        rfe = RFE(lin_reg, n_col)
        fit = rfe.fit(train_x, train_y)
    
        # Selecte most relevant attributes for machien learning
        fit_list = fit.support_.tolist()
        indexes = [index for index in range(len(fit_list)) if fit_list[index] == True]
    
        # Print out attributes selected and ranking
        print('\nAttributes selected are: ', itemgetter(*indexes)(attr_list))
        print('\nAttributes Ranking: ', fit.ranking_)

        train_x_returned = train_x.iloc[:,indexes]
        test_x_returned = test_x.iloc[:,indexes]
    
    # Using factor analysis
    elif method == 'fact':
        fact_anal = FactorAnalysis(n_components=n_col)
        train_x_returned = pd.DataFrame(fact_anal.fit_transform(train_x))
        test_x_returned = pd.DataFrame(fact_anal.transform(test_x))
    
        train_x_returned.columns = [''.join(['feature_',str(i)]) for i in list(train_x_returned.columns)]
        test_x_returned.columns = [''.join(['feature_', str(i)]) for i in list(test_x_returned.columns)]
    
    # Using PCA
    elif method == 'PCA':
        pca_down = PCA(n_components=n_col)
        train_x_returned = pd.DataFrame(pca_down.fit_transform(train_x))
        test_x_returned = pd.DataFrame(pca_down.transform(test_x))
    
        train_x_returned.columns = [''.join(['feature_',str(i)]) for i in list(train_x_returned.columns)]
        test_x_returned.columns = [''.join(['feature_', str(i)]) for i in list(test_x_returned.columns)]
    
    # Returned selected or regenerated features
    return train_x_returned, test_x_returned
    
main()