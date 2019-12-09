# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:38:46 2019

@author: Jimny
"""

import tensorflow as tf

## Import needed packages
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

from sklearn.neural_network import MLPClassifier
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

from sklearn.metrics import mutual_info_score,roc_auc_score,confusion_matrix,accuracy_score, roc_curve, auc, classification_report

from scipy import interp

# example of dropout between fully connected layers
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import plot_model

from numpy.random import seed
seed(117)

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
  
    # Define the neural network model with dropout layer after each dense layer
    model = Sequential()
    model.add(Dense(40, input_dim = len(X_train.columns), activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(20,activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fit the neural network model
    fitted = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose = 1)
    
    # Evaluate the neural network model on train and test dataset
    _ , train_acc = model.evaluate(X_train, y_train, verbose=0)
    _ , test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    # Plot train and test accuracy and loss history versus number of epoches
    # Set plot parameters
    plt.style.use('ggplot')
    params = {'legend.fontsize': 16,
              'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.rcParams['axes.titlepad'] = 20 
    matplotlib.rc('xtick',labelsize=14)
    matplotlib.rc('ytick',labelsize=14)
    
    # Plot train and test accuracy vs. number of epoches
    plt.figure(figsize=(8,7))
    plt.plot(fitted.history['accuracy'], label='train', linewidth = 3)
    plt.plot(fitted.history['val_accuracy'], label='test', linewidth = 3)
    plt.xlabel('Number of Iteration', size = 16)
    plt.ylabel('Acccuracy', size = 16)
    plt.legend(prop={'size': 6})
    plt.ylim([0.6,1])
    plt.legend()
    plt.title('First Hidden Layer: 12 Neurons [Drop Out Rate = 0]\nSecond Hidden Layer: 6 Neurons [Drop Out Rate = 0]',
              size = 18)
    #plt.tight_layout()
    plt.show()
    
    # Plot train and test loss vs. number of epoches
    plt.figure(figsize=(8,7))
    plt.plot(fitted.history['loss'], label='train', linewidth = 3)
    plt.plot(fitted.history['val_loss'], label='test', linewidth = 3)
    plt.xlabel('Number of Iteration', size = 16)
    plt.ylabel('Loss', size = 16)
    plt.legend(prop={'size': 6})
    plt.ylim([0,1])
    plt.legend()
    plt.title('First Hidden Layer: 12 Neurons [Drop Out Rate = 0]\nSecond Hidden Layer: 6 Neurons [Drop Out Rate = 0]',
              size = 18)
    #plt.tight_layout()
    plt.show()
    
    #plot_model(model, to_file='model.png')
    
    # Make prediction on train and test data
    prediction = model.predict(X_test)[:,0]
    pred_train = model.predict(X_train)[:,0]
    
    # Set threshold and make final prediction
    thres = 0.5
    pred_drop = pd.Series(np.full(len(y_test), 0)) 
    pred_drop[prediction >= thres] = 1
    
    # Map response into integers
    y_test = list(map(int, y_test))
    y_train = list(map(int, y_train))
    
    print_validate(y_test, pred_drop, 'Drop Out', 
                   colors = "Greens",v_range = [0,800])
    
    # ROC Curves for test dataset
    plt.figure(figsize=(8,7))
    draw_roc(y_test, prediction, 'Test Data', 'royalblue', '-')
    draw_roc(y_train, pred_train, 'Train Data', 'lightcoral', '--')
    
    plt.plot([0, 1], [0, 1], 'k--', lw = 4)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Test Result')
    plt.legend(loc="lower right", fontsize = 14, handlelength=4)
    plt.show()
    
# This function is used to print validation/test resuls
def print_validate(Y_val, predictions, plt_title, 
                   colors = "Blues", v_range = [0,1000]):
    conf_mat = confusion_matrix(Y_val, predictions)
    print("\nAccuracy score: ",accuracy_score(Y_val, predictions))
    print("\nConfusion Matrix: \n",conf_mat)
    print(classification_report(Y_val, predictions))

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(predictions, Y_val,
                          title='Confusion Matrix of '+plt_title,
                          col_map = colors, v_range = v_range)

def plot_confusion_matrix(y_pred, y_true, title, col_map = "Blues", 
                          v_range = [0,1000]):
    conf_mat = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(conf_mat, columns=['Non-Fatal','Fatal'], index = ['Non-Fatal','Fatal'])
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (6,5))
    plt.title(title, fontsize = 16)
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap=col_map, annot=True,annot_kws={"size": 16}, fmt='g',
                vmin=v_range[0], vmax=v_range[1])# font size
  
# This function is used to draw ROC curves
def draw_roc(y_true, y_proba, clf_name, col, style):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr,  lw=4, color = col, linestyle = style,
             label='ROC of {0} (AUC = {1:0.2f})'
                 ''.format(clf_name, roc_auc))


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

