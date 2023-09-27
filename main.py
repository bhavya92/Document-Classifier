
import spacy
import json 
import csv
import pickle
import pandas as pd
from pprint import pprint
import time
import seaborn as sns
import altair as alt
import altair_viewer as av

import numpy as np
import os
import sys

from matplotlib import pyplot as plt
from collections import Counter

from nltk.corpus import stopwords

from sklearn import metrics 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from keras.layers import Dropout, Dense
from keras.models import Sequential

np.set_printoptions(threshold=sys.maxsize)

sns.set_style("whitegrid")

nlp = spacy.load("en_core_web_sm")

nlp.max_length = 3000000

stop_words = set(stopwords.words('english'))

symbols = ["`","~","!","@","#",'$','%',"^","&","*","(",")","-","_","=","+","[","]","{","}","\\","|",";",":",'"',"'",'""',"''",",","<",".",">","/","?","--"," ","’s","’"]

def visualize_number(df):
    bars = alt.Chart(df).mark_bar(size=100).encode(
        x = alt.X("Category"),
        y = alt.Y("count():Q",axis=alt.Axis(title='Number of Documents')),
        tooltip=[alt.Tooltip('count()',title='Number of Documents'),'Category'],
        color = 'Category'
    )
    text = bars.mark_text(
        align='center',
        baseline='bottom',
    ).encode(
        text='count()'
    )
    #av.show(
    (bars + text).interactive().properties(
        height = 300,
        width = 700,
        title = 'Number of Documents in each Category'
    )
    #)

def visualize_percentage(df):
    df2 = pd.DataFrame(df.groupby('Category').count()['Id']).reset_index()
    print(df2.head)
    bars = alt.Chart(df2).mark_bar(size=100).encode(
        x = alt.X("Category"),
        y = alt.Y("PercentofTotal:Q",axis=alt.Axis(format='.0%',title="% of Documents")),
        color = "Category"
    ).transform_window(
        TotalArticles='sum(Id)',
        frame=[None,None]
    ).transform_calculate(
        PercentofTotal="datum.Id / datum.TotalArticles"
    )

    text = bars.mark_text(
        align = 'center',
        baseline='bottom'
    ).encode(
        text = alt.Text('PercentofTotal:Q',format='.1%')
    )

    #av.show(
    (bars + text).interactive().properties(
            height = 300,
            width = 700,
            title = " Percentage of Documents in Each Category"
    )
    #)

def visualize_document_length(df):
    plt.figure(figsize=(12.8,6))
    sns.distplot(df['text_length']).set_title('Article length distribution')
    #plt.show()

def plot_dim_red(model, features, labels, n_components=2):
    
    # Creation of the model
    if (model == 'PCA'):
        mod = PCA(n_components=n_components)
        title = "PCA decomposition"  # for the plot
        
    elif (model == 'TSNE'):
        mod = TSNE(n_components=2)
        title = "t-SNE decomposition" 

    else:
        return "Error"
    
    # Fit and transform the features
    principal_components = mod.fit_transform(features)
    
    # Put them into a dataframe
    df_features = pd.DataFrame(data=principal_components,
                     columns=['PC1', 'PC2'])
    
    # Now we have to paste each row's label and its meaning
    # Convert labels array to df
    df_labels = pd.DataFrame(data=labels,
                             columns=['label'])
    
    df_full = pd.concat([df_features, df_labels], axis=1)
    df_full['label'] = df_full['label'].astype(str)

    # Get labels name
    category_names = {
        "0": 'Wikileaks',
        "1": 'Business',
        "2": 'Sports'
    }

    # And map labels
    df_full['label_name'] = df_full['label']
    df_full = df_full.replace({'label_name':category_names})

    # Plot
    plt.figure(figsize=(10,10))
    sns.scatterplot(x='PC1',
                    y='PC2',
                    hue="label_name", 
                    data=df_full,
                    palette=["red", "royalblue", "lightseagreen"],
                    alpha=.7).set_title(title);
    plt.show()

def load_data():
    texts = []
    labels = []
    filename = []
    category = []
    for c in ['more_sensitive','sensitive','less_sensitive']:
        train_path = os.path.join('data',c)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path,fname),encoding='utf-8') as f:
                    text = f.read()
                    # print(type(text))
                    #text = text.decode('utf-8')
                    texts.append(text)
                    filename.append(fname)
                #more_sensitive == 0, sensitive == 1, less_sensitive == 2     
                if c == 'more_sensitive':
                    labels.append(0)
                    category.append('Wikileaks')
                elif c == 'sensitive':
                    category.append('Buisness')
                    labels.append(1)
                else:
                    category.append('Sports')
                    labels.append(2)    

    temp = {'Name' : filename, 'Content' : texts, 'Category' : category,'Id' : labels}

    df = pd.DataFrame(temp)
   

    """
    Visualizing Number of Documents in Each Category
    Uncomment below to Visualize
    """
    #visualize_number(df)

    """
    Visualizing percentage of Documents in each Category
    Uncomment below to Visualize
    """
    #visualize_percentage(df)
   
    
    #Getting length of each Article and adding to the dataframe
    df['text_length'] = df['Content'].str.len()
    
    """
    Visualizing Document Length by Category
    Uncomment below to Visualize
    """
    #visualize_document_length(df)
    
    #Removing Documents with more than 25000 words for better results 
    df_25k = df[df['text_length'] < 25000] 

    """
    Visualizing new Data Frame 
    Uncomment below to Visualize
    """
    
    """
    In the final dataset, we can see that artciles of more_sensitive category have more number of words per article,
    but this will not affect the final results as we will normalize the features while creating Tf-Idf Scores
    """

    visualize_number(df_25k)
    visualize_percentage(df_25k)
    visualize_document_length(df_25k)

    #Saving the new dataset
    with open('Pickle/Sensitivity_datatset.pickle','wb') as output:
        pickle.dump(df_25k,output)


def preprocess():
    
    """
    Load the dataset
    """
    
    with open('Pickle/Sensitivity_datatset.pickle','rb') as data:
        df = pickle.load(data)

    """
    Replacing Next Line escape character with Space 
    """
    df['Content_NoEC'] = df['Content'].str.replace("\n"," ")
    

    """
    Converting text to lower case 
    """
    df['Content_lowerCase'] = df['Content_NoEC'].str.lower()

    """
    Here we will be convertig the text in to tokens using SpaCy Library. 
    We will check the Part-Of-Speech(POS) tagging of every word provided by Spacy and 
    filter the words accordingly.

    We will remove words with POS tagging of SYM(Symbol), NUM(Number), X(Not identified by Spacy), PART(Participles), PUNCT(Punctuation) and SPACE(Spaces)
    For better results and more accuracy we will check the word's lemma   to remove any symbol present in symbols list declared above and also any word in stop_words provied by NLTK.
    Finally we will use word's lemma form (Lemmatizing) using SpaCy's token.lemma_
    """

    #Tokenizing , Lemmatizing and Filtering
    df['Content_tokenize'] = df['Content_lowerCase']
    tokenize_content = list(df['Content_tokenize'].values)
    
    #Empty list to append processed text 
    result = []
    
    for value in tokenize_content:
        text = '' 
        doc = nlp(value)

        for token in doc:
            if(token.pos_ != 'SYM' and token.pos_ != 'NUM' and token.pos_ != 'X' and token.pos_ != 'PART' and token.pos_ != 'SPACE' and token.tag_ != 'XX' and token.pos_ != 'PUNCT' and token.lemma_ != '-PRON-'):
                if(token.lemma_ not in symbols and token.lemma_ not in stop_words):
                    text = text + token.lemma_ + " "

        #This part is to remove symbols which may be inside a word (this can happen mostly in wikileaks dataset because we extracted text from PDFs using tesseract-ocr)
        for symbol in symbols:
            text = text.replace(symbol,' ')
        
        result.append(text)

    df['Content_Processed'] = result
    
    list_columns = ['Name','Content','Category','Id','Content_Processed']
    df = df[list_columns]
   
    #Splitting Data for Training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(df['Content_Processed'],df['Id'],test_size = 0.2,random_state = 8)

    ngram = (1,2)
    min_df = 10
    max_df = 1.
    max_features = 300

    tfidf = TfidfVectorizer(encoding='utf-8',ngram_range=ngram,stop_words=None, lowercase= False, max_df=max_df, min_df=min_df, max_features=max_features, norm='l2',sublinear_tf=True)

    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = Y_train
    
    #print(features_train)
    
    features_test = tfidf.transform(X_test).toarray()
    labels_test = Y_test
    
    category_codes = {
        'wikileaks' : 0,
        'business' : 1,
        'sports' : 2
    }


    """
    This is used to find most correlated the unigrams and bigrams (Used in TfidfVectorizer) using chi-squared
    """
    for Product, category in sorted(category_codes.items()):
        features_chi2 = chi2(features_train,labels_train == category)
        indices = np.argsort(features_chi2[0])
        feature_name = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_name if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_name if len(v.split(' ')) == 2]
        print("# '{}' category:".format(Product))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
        print("")        

    #Saving X_train
    with open('Pickle/X_train.pickle','wb') as f:
        pickle.dump(X_train,f)    

    #Saving X_test
    with open('Pickle/X_test.pickle','wb') as f:
        pickle.dump(X_test,f)    

    #Saving Y_train
    with open('Pickle/Y_train.pickle','wb') as f:
        pickle.dump(Y_train,f)    

    #Saving Y_test
    with open('Pickle/Y_test.pickle','wb') as f:
        pickle.dump(Y_test,f)        

    #Saving new df
    with open('Pickle/processed_df.pickle','wb') as f:
        pickle.dump(df,f)        

    #Saving features_train
    with open('Pickle/features_train.pickle','wb') as f:
        pickle.dump(features_train,f)        

    #Saving labels_train
    with open('Pickle/labels_train.pickle', 'wb') as f:
        pickle.dump(labels_train, f)

    #Saving features_test
    with open('Pickle/features_test.pickle', 'wb') as f:
        pickle.dump(features_test, f)

    #Saving labels_test
    with open('Pickle/labels_test.pickle', 'wb') as f:
        pickle.dump(labels_test, f)
        
    #Saving TF-IDF object
    with open('Pickle/tfidf.pickle', 'wb') as f:
        pickle.dump(tfidf, f)

def dimension_reduction():
    #Loading the data
    
    #Processed Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    #features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train,'rb') as data:
        features_train = pickle.load(data)

    #labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train,'rb') as data:
        labels_train = pickle.load(data)

    #features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test,'rb') as data:
        features_test = pickle.load(data)

    #labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test,'rb') as data:
        labels_test = pickle.load(data)

    print(features_train.shape)
    print(features_test.shape)
    print(labels_train.shape)
    print(labels_test.shape)   

    features = np.concatenate((features_train,features_test), axis=0)
    labels = np.concatenate((labels_train,labels_test), axis=0)     

    plot_dim_red("PCA", features=features, labels=labels,n_components=2)

    plot_dim_red("TSNE", features=features, labels=labels, n_components=2)




def svm_model_training():
    
    #Loading the data
    
    #Processed Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    #features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train,'rb') as data:
        features_train = pickle.load(data)

    #labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train,'rb') as data:
        labels_train = pickle.load(data)

    #features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test,'rb') as data:
        features_test = pickle.load(data)

    #labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test,'rb') as data:
        labels_test = pickle.load(data)

    print(features_train.shape)
    print(features_test.shape)

    """
    Here we will use Randomize Search Cross Validation for Hyperparameter tuning
    """

    _svc = svm.SVC(random_state=8)

    print("\nParameters in use:\n")
    pprint(_svc.get_params())

    #Randomized Search Cross Validation
    C = [.0001, .001, .01]
    gamma = [.0001, .001, .01, .1, 1, 10, 100]
    degree = [1,2,3,4,5]
    kernel = ['linear','rbf','poly']
    probab = [True]
    random_grid = {
        'C' : C,
        'kernel' : kernel,
        'gamma' : gamma,
        'degree' : degree,
        'probability' : probab
    }

    pprint(random_grid)

    #Base Model to tune
    svc = svm.SVC(random_state=8)

    random_search = RandomizedSearchCV(estimator = svc,param_distributions = random_grid,n_iter=50,scoring='accuracy',cv=3,verbose=1,random_state=8)

    random_search.fit(features_train,labels_train)

    print("\n\n")
    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    """
    Now we will do more search centred around those values using Grid Search Cross Validation 
    """

    #Parameter grid based on above results
    C = [.0001, .001, .01, .1]
    degree = [3,4,5]
    gamma = [1,10,100]
    probab = [True]

    param_grid = [
        {'C' : C, 'kernel' : ['linear'], 'probability' : probab},
        {'C' : C, 'kernel' : ['poly'], 'probability' : probab},
        {'C' : C, 'kernel' : ['rbf'], 'probability' : probab}
    ]

    # Create a base model
    svc = svm.SVC(random_state=8)

    cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

    grid_search = GridSearchCV(estimator = svc, param_grid =param_grid, scoring='accuracy',cv = cv_sets, verbose= 1)

    grid_search.fit(features_train,labels_train)


    print("\n\nThe best hyperparameters from Grid Search are:")
    print(grid_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(grid_search.best_score_)

    best_svc = grid_search.best_estimator_

    #Now we will fit the training data in our model

    best_svc.fit(features_train,labels_train)

    svc_pred = best_svc.predict(features_test)

    #Training Accuracy
    print("\nThe training accuracy is: ")
    print(accuracy_score(labels_train, best_svc.predict(features_train))) 

    # Test accuracy
    print("\nThe test accuracy is: ")
    print(accuracy_score(labels_test, svc_pred))

    # Classification report 
    print("\nClassification report")
    print(classification_report(labels_test,svc_pred))

    #Confusion Matrix
    
    aux_df = df[['Category', 'Id']].drop_duplicates().sort_values('Id')
    conf_matrix = confusion_matrix(labels_test, svc_pred)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df['Category'].values, 
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    #plt.show()    

    d = {
     'Model': 'SVM',
     'Training Set Accuracy': accuracy_score(labels_train, best_svc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, svc_pred)
    }

    df_models_svc = pd.DataFrame(d, index=[0])

    print(df_models_svc)

    with open('Pickle/Models/best_svc.pickle', 'wb') as output:
        pickle.dump(best_svc, output)
    
    with open('Pickle/Models/df_models_svc.pickle', 'wb') as output:
        pickle.dump(df_models_svc, output)


def knn_model_training():
    #Loading the data
    
    #Processed Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    #features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train,'rb') as data:
        features_train = pickle.load(data)

    #labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train,'rb') as data:
        labels_train = pickle.load(data)

    #features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test,'rb') as data:
        features_test = pickle.load(data)

    #labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test,'rb') as data:
        labels_test = pickle.load(data)

    """
    Cross Validation for Hyperparameter tuning
    """    

    _knn =  KNeighborsClassifier()
    print('\n\nParameters currently in use:\n')
    pprint(_knn.get_params())

    #Grid Search Cross Validation
    n_neighbors =  [int(x) for x in np.linspace(start = 1, stop = 500, num = 100)]

    param_grid = {'n_neighbors' : n_neighbors}

    #Creating base model
    knn = KNeighborsClassifier()

    cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=cv_sets, verbose=1)    

    grid_search.fit(features_train,labels_train)

    print("\nThe best hyperparameters from Grid Search are:")
    print(grid_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(grid_search.best_score_)

    best_knn = grid_search.best_estimator_
    print("\n\nBest KNN Model")
    print(best_knn)

    #Fitting data in the KNN Model
    best_knn.fit(features_train, labels_train)

    knn_pred = best_knn.predict(features_test)

    # Training accuracy
    print("\nThe training accuracy is: ")
    print(accuracy_score(labels_train, best_knn.predict(features_train)))

    # Test accuracy
    print("\nThe test accuracy is: ")
    print(accuracy_score(labels_test, knn_pred))

    # Classification report
    print("\nClassification report")
    print(classification_report(labels_test,knn_pred))

    #Confusion Matrix
    aux_df = df[['Category', 'Id']].drop_duplicates().sort_values('Id')
    conf_matrix = confusion_matrix(labels_test, knn_pred)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df['Category'].values, 
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    #plt.show()

    d = {
        'Model': 'KNN',
        'Training Set Accuracy': accuracy_score(labels_train, best_knn.predict(features_train)),
        'Test Set Accuracy': accuracy_score(labels_test, knn_pred)
    }

    df_models_knn = pd.DataFrame(d, index=[0])

    print("\n\n")
    print(df_models_knn)

    with open('Pickle/Models/best_knn.pickle', 'wb') as output:
        pickle.dump(best_knn, output)

    with open('Pickle/Models/df_models_knn.pickle', 'wb') as output:
        pickle.dump(df_models_knn, output)

def mlr_model_training():
    #Loading the data
    
    #Processed Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    #features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train,'rb') as data:
        features_train = pickle.load(data)

    #labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train,'rb') as data:
        labels_train = pickle.load(data)

    #features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test,'rb') as data:
        features_test = pickle.load(data)

    #labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test,'rb') as data:
        labels_test = pickle.load(data)

    """
    Cross Validation for Hyperparameter tuning
    """    
    _lr = LogisticRegression(random_state = 8)

    print('\nParameters currently in use:\n')
    pprint(_lr.get_params())

    #Randomized Search Cross Validation
    C = [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)]

    #multi_class
    multi_class = ['multinomial']

    #solver
    solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
    
    #class_weight
    class_weight = ['balanced', None]

    #penalty
    penalty = ['l2']

    #Creating the random grid
    random_grid = {'C': C,
                'multi_class': multi_class,
                'solver': solver,
                'class_weight': class_weight,
                'penalty': penalty}

    pprint(random_grid)

    #Creating base model
    lr = LogisticRegression(random_state = 8)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=lr,
                                    param_distributions=random_grid,
                                    n_iter=50,
                                    scoring='accuracy',
                                    cv=3, 
                                    verbose=1, 
                                    random_state=8)

    # Fit the random search model
    random_search.fit(features_train, labels_train)

    print("\nThe best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    #Grid Search Cross Validation
    # Create the parameter grid based on the results of random search 
    C = [float(x) for x in np.linspace(start = 0.6, stop = 1, num = 10)]
    multi_class = ['multinomial']
    solver = ['sag']
    class_weight = ['balanced']
    penalty = ['l2']

    param_grid = {'C': C,
                'multi_class': multi_class,
                'solver': solver,
                'class_weight': class_weight,
                'penalty': penalty}

    # Create a base model
    lr = LogisticRegression(random_state=8)

    # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
    cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=lr, 
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=cv_sets,
                            verbose=1)

    # Fit the grid search to the data
    grid_search.fit(features_train, labels_train)

    print("\nThe best hyperparameters from Grid Search are:")
    print(grid_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(grid_search.best_score_)

    best_lr = grid_search.best_estimator_

    print(best_lr)

    #Fitting our data in lr model
    best_lr.fit(features_train, labels_train)
    lr_pred = best_lr.predict(features_test)

    # Training accuracy
    print("\nThe training accuracy is: ")
    print(accuracy_score(labels_train, best_lr.predict(features_train)))    

    # Test accuracy
    print("\nThe test accuracy is: ")
    print(accuracy_score(labels_test, lr_pred))

    # Classification report
    print("\nClassification report")
    print(classification_report(labels_test,lr_pred))

    #Confusion Matrix
    aux_df = df[['Category', 'Id']].drop_duplicates().sort_values('Id')
    conf_matrix = confusion_matrix(labels_test, lr_pred)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df['Category'].values, 
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    #plt.show()

    d = {
     'Model': 'Logistic Regression',
     'Training Set Accuracy': accuracy_score(labels_train, best_lr.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, lr_pred)
    }

    df_models_lr = pd.DataFrame(d, index=[0])    
    print("\n\n")
    print(df_models_lr)

    with open('Pickle/Models/best_lr.pickle', 'wb') as output:
        pickle.dump(best_lr, output)
        
    with open('Pickle/Models/df_models_lr.pickle', 'wb') as output:
        pickle.dump(df_models_lr, output)


def mnb_model_training():
    #Loading the data
    
    #Processed Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    #features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train,'rb') as data:
        features_train = pickle.load(data)

    #labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train,'rb') as data:
        labels_train = pickle.load(data)

    #features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test,'rb') as data:
        features_test = pickle.load(data)

    #labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test,'rb') as data:
        labels_test = pickle.load(data)

    mnb = MultinomialNB()
    print("\n")
    print(mnb)

    #Fitting data in model
    mnb.fit(features_train,labels_train)
    mnb_pred = mnb.predict(features_test)

    # Training accuracy
    print("\nThe training accuracy is: ")
    print(accuracy_score(labels_train, mnb.predict(features_train)))

    # Test accuracy
    print("\nThe test accuracy is: ")
    print(accuracy_score(labels_test, mnb_pred))

    # Classification report
    print("\nClassification report")
    print(classification_report(labels_test,mnb_pred))

    #Confusion Matrix
    aux_df = df[['Category', 'Id']].drop_duplicates().sort_values('Id')
    conf_matrix = confusion_matrix(labels_test, mnb_pred)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df['Category'].values, 
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

    d = {
     'Model': 'Multinomial Naïve Bayes',
     'Training Set Accuracy': accuracy_score(labels_train, mnb.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, mnb_pred)
    }

    df_models_mnb = pd.DataFrame(d, index=[0])    

    print("\n")
    print(df_models_mnb)

    with open('Pickle/Models/best_mnb.pickle', 'wb') as output:
        pickle.dump(mnb, output)
    
    with open('Pickle/Models/df_models_mnb.pickle', 'wb') as output:
        pickle.dump(df_models_mnb, output)


def rf_model_training():
    #Loading the data
    
    #Processed Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    #features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train,'rb') as data:
        features_train = pickle.load(data)

    #labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train,'rb') as data:
        labels_train = pickle.load(data)

    #features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test,'rb') as data:
        features_test = pickle.load(data)

    #labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test,'rb') as data:
        labels_test = pickle.load(data)

    """
    Cross Validation for Hyperparameter tuning
    """
    _rf = RandomForestClassifier(random_state = 8)
    print('\nParameters currently in use:\n')
    pprint(_rf.get_params())
    print("\n")

    #Randomized Search Cross Validation
    
    #n_estimators
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

    #max_features
    max_features = ['auto', 'sqrt']

    #max_depth
    max_depth = [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)]
    #max_depth.append(None)

    #min_samples_split
    min_samples_split = [2, 5, 10]

    #min_samples_leaf
    min_samples_leaf = [1, 2, 4]

    #bootstrap
    bootstrap = [True, False]

    #Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    pprint(random_grid)

    #Base Model
    rf = RandomForestClassifier(random_state=8)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, scoring='accuracy', cv=3,  verbose=1, random_state=8)

    #Fit the random search model
    random_search.fit(features_train, labels_train)

    print("\nThe best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("\nThe mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)        

    #Grid Search Cross Validation
    bootstrap = [False]
    max_depth = [30, 40, 50]
    max_features = ['sqrt']
    min_samples_leaf = [1, 2, 4]
    min_samples_split = [5, 10, 15]
    n_estimators = [800]

    param_grid = {
        'bootstrap': bootstrap,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators
    }

    # Create a base model
    rf = RandomForestClassifier(random_state=8)

    # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
    cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, 
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=cv_sets,
                            verbose=1)

    # Fit the grid search to the data
    grid_search.fit(features_train, labels_train)

    print("\nThe best hyperparameters from Grid Search are:")
    print(grid_search.best_params_)
    print("")
    print("\nThe mean accuracy of a model with these hyperparameters is:")
    print(grid_search.best_score_)

    best_rf = grid_search.best_estimator_
    print("\n\n")
    print(best_rf)

    #Fitting data into the model
    best_rf.fit(features_train, labels_train)
    rf_pred = best_rf.predict(features_test)

    # Training accuracy
    print("\nThe training accuracy is: ")
    print(accuracy_score(labels_train, best_rf.predict(features_train)))

    # Test accuracy
    print("\nThe test accuracy is: ")
    print(accuracy_score(labels_test, rf_pred))

    # Classification report
    print("\nClassification report")
    print(classification_report(labels_test,rf_pred))

    aux_df = df[['Category', 'Id']].drop_duplicates().sort_values('Id')
    conf_matrix = confusion_matrix(labels_test, rf_pred)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df['Category'].values, 
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

    d = {
     'Model': 'Random Forest',
     'Training Set Accuracy': accuracy_score(labels_train, best_rf.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, rf_pred)
    }

    df_models_rf = pd.DataFrame(d, index=[0])

    print("\n")
    print(df_models_rf)

    with open('Pickle/Models/best_rf.pickle', 'wb') as output:
        pickle.dump(best_rf, output)
    
    with open('Pickle/Models/df_models_rf.pickle', 'wb') as output:
        pickle.dump(df_models_rf, output)

def gb_model_training():
    #Loading the data
    
    #Processed Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    #features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train,'rb') as data:
        features_train = pickle.load(data)

    #labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train,'rb') as data:
        labels_train = pickle.load(data)

    #features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test,'rb') as data:
        features_test = pickle.load(data)

    #labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test,'rb') as data:
        labels_test = pickle.load(data)

    _gb = GradientBoostingClassifier(random_state = 8)

    print('\nParameters currently in use:\n')
    pprint(_gb.get_params())

    #n_estimators
    n_estimators = [200, 800]

    #max_features
    max_features = ['auto', 'sqrt']

    #max_depth
    max_depth = [10, 40]
    max_depth.append(None)

    #min_samples_split
    min_samples_split = [10, 30, 50]

    #min_samples_leaf
    min_samples_leaf = [1, 2, 4]

    #learning rate
    learning_rate = [.1, .5]

    #subsample
    subsample = [.5, 1.]

    #Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'learning_rate': learning_rate,
                'subsample': subsample}

    pprint(random_grid)
    
    gb = GradientBoostingClassifier(random_state=8)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=gb,
                                    param_distributions=random_grid,
                                    n_iter=50,
                                    scoring='accuracy',
                                    cv=3, 
                                    verbose=1, 
                                    random_state=8)

    # Fit the random search model
    random_search.fit(features_train, labels_train)

    print("\nThe best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("\nThe mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    #Grid Search Cross Validation
    max_depth = [5, 10, 15]
    max_features = ['sqrt']
    min_samples_leaf = [2]
    min_samples_split = [50, 100]
    n_estimators = [800]
    learning_rate = [.1, .5]
    subsample = [1.]

    param_grid = {
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'subsample': subsample

    }

    # Create a base model
    gb = GradientBoostingClassifier(random_state=8)

    # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
    cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=gb, 
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=cv_sets,
                            verbose=1)

    # Fit the grid search to the data
    grid_search.fit(features_train, labels_train)

    print("\nThe best hyperparameters from Grid Search are:")
    print(grid_search.best_params_)
    print("")
    print("\nThe mean accuracy of a model with these hyperparameters is:")
    print(grid_search.best_score_)

    best_gb = grid_search.best_estimator_
    print(best_gb)

    best_gb.fit(features_train, labels_train)
    gb_pred = best_gb.predict(features_test)

    # Training accuracy
    print("\nThe training accuracy is: ")
    print(accuracy_score(labels_train, best_gb.predict(features_train)))    

    # Test accuracy
    print("\nThe test accuracy is: ")
    print(accuracy_score(labels_test, gb_pred))

    # Classification report
    print("\nClassification report")
    print(classification_report(labels_test,gb_pred))

    #Confusion_matrix
    aux_df = df[['Category', 'Id']].drop_duplicates().sort_values('Id')
    conf_matrix = confusion_matrix(labels_test, gb_pred)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df['Category'].values, 
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

    d = {
     'Model': 'Gradient Boosting',
     'Training Set Accuracy': accuracy_score(labels_train, best_gb.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, gb_pred)
    }

    df_models_gb = pd.DataFrame(d, index=[0])

    print(df_models_gb)

    with open('Pickle/Models/best_gbc.pickle', 'wb') as output:
        pickle.dump(best_gb, output)
    
    with open('Pickle/Models/df_models_gbc.pickle', 'wb') as output:
        pickle.dump(df_models_gb, output)

def build_dnn_model(shape, no_class, dropout=0.2):
    model = Sequential()
    node = 300
    no_layers = 3

    model.add(Dense(node,input_dim = shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,no_layers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(no_class, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

def dnn_model_training():
    #Loading the data
    
    #Processed Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    #features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train,'rb') as data:
        features_train = pickle.load(data)

    #labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train,'rb') as data:
        labels_train = pickle.load(data)

    #features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test,'rb') as data:
        features_test = pickle.load(data)

    #labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test,'rb') as data:
        labels_test = pickle.load(data)

    # X_train
    path_X_train = "Pickle/X_train.pickle"
    with open(path_X_train, 'rb') as data:
        X_train = pickle.load(data)

    # X_test
    path_X_test = "Pickle/X_test.pickle"
    with open(path_X_test, 'rb') as data:
        X_test = pickle.load(data)

    # y_train
    path_y_train = "Pickle/Y_train.pickle"
    with open(path_y_train, 'rb') as data:
        y_train = pickle.load(data)

    # y_test
    path_y_test = "Pickle/Y_test.pickle"
    with open(path_y_test, 'rb') as data:
        y_test = pickle.load(data)

    model_DNN = build_dnn_model(features_train.shape[1],3)

    model_DNN.fit(features_train, labels_train, validation_data=(features_test,labels_test),epochs=5,batch_size=128,verbose=2)

    predicted = model_DNN.predict_classes(features_test) 
    predicted2 = model_DNN.predict_classes(features_train)

    w,b,s = 0,0,0

    for val in predicted:
        if val == 0:
            w = w+1
        elif val == 1:
            b = b + 1
        else:
            s = s + 1         

    for val in predicted2:
        if val == 0:
            w = w+1
        elif val == 1:
            b = b + 1
        else:
            s = s + 1 

    print("W" + str(w))
    print("B" + str(b))
    print("S" + str(s))


    print(len(predicted))
    print(classification_report(labels_test,predicted))
    x = labels_test.to_numpy()
    print(x)
    #Confusion_matrix
    aux_df = df[['Category', 'Id']].drop_duplicates().sort_values('Id')

    conf_matrix = confusion_matrix(labels_test, predicted)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df['Category'].values, 
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

    right = 0
    wrong = 0
    for i in range(0,len(predicted)):
        if(predicted[i] == x[i]):
            right = right + 1
        else:
            wrong = wrong + 1  

    print("Right prediction : " + str( (right/(right+wrong)) * 100 ) + "%" )
    print("Wrong prediction : " + str( (wrong/(right+wrong)) * 100 ) + "%" )



def model_results():

    path = "Pickle/Models/"

    list_pickles = ["df_models_knn.pickle","df_models_lr.pickle","df_models_svc.pickle"]

    df_summary = pd.DataFrame()

    for pickle_ in list_pickles:
        temp_path = path
        temp_path = temp_path + pickle_
        with open(temp_path, 'rb') as data:
            df = pickle.load(data)

        df_summary = df_summary.append(df)

    df_summary = df_summary.reset_index().drop('index', axis=1)

    print(df_summary.sort_values('Test Set Accuracy', ascending=False))

    # Dataframe
    path_df = "Pickle/processed_df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)
        
    # X_train
    path_X_train = "Pickle/X_train.pickle"
    with open(path_X_train, 'rb') as data:
        X_train = pickle.load(data)

    # X_test
    path_X_test = "Pickle/X_test.pickle"
    with open(path_X_test, 'rb') as data:
        X_test = pickle.load(data)

    # y_train
    path_y_train = "Pickle/Y_train.pickle"
    with open(path_y_train, 'rb') as data:
        y_train = pickle.load(data)

    # y_test
    path_y_test = "Pickle/Y_test.pickle"
    with open(path_y_test, 'rb') as data:
        y_test = pickle.load(data)

    # features_train
    path_features_train = "Pickle/features_train.pickle"
    with open(path_features_train, 'rb') as data:
        features_train = pickle.load(data)

    # labels_train
    path_labels_train = "Pickle/labels_train.pickle"
    with open(path_labels_train, 'rb') as data:
        labels_train = pickle.load(data)

    # features_test
    path_features_test = "Pickle/features_test.pickle"
    with open(path_features_test, 'rb') as data:
        features_test = pickle.load(data)

    # labels_test
    path_labels_test = "Pickle/labels_test.pickle"
    with open(path_labels_test, 'rb') as data:
        labels_test = pickle.load(data)
        
    # SVM Model
    path_model = "Pickle/Models/best_svc.pickle"
    with open(path_model, 'rb') as data:
        svc_model = pickle.load(data)
        
    # LR Model
    path_model = "Pickle/Models/best_lr.pickle"
    with open(path_model, 'rb') as data:
        lr_model = pickle.load(data)    

    # KNN Model
    path_model = "Pickle/Models/best_knn.pickle"
    with open(path_model, 'rb') as data:
        knn_model = pickle.load(data)
    
    # RF Model
    path_model = "Pickle/Models/best_rf.pickle"
    with open(path_model, 'rb') as data:
        rf_model = pickle.load(data)
    
    # MNB Model
    path_model = "Pickle/Models/best_mnb.pickle"
    with open(path_model, 'rb') as data:
        mnb_model = pickle.load(data)

    # GB Model
    path_model = "Pickle/Models/best_gbc.pickle"
    with open(path_model, 'rb') as data:
        gb_model = pickle.load(data)    

    predictions1 = gb_model.predict(features_train)
    predictions2 = gb_model.predict(features_test)
    s = []
    b = []
    w = []
    for code in predictions1:
        if(code == 0):
            w.append(code)
        elif(code == 1):
            b.append(code)
        else:
            s.append(code)        

    for code in predictions2:
        if(code == 0):
            w.append(code)
        elif(code == 1):
            b.append(code)
        else:
            s.append(code)        

    print("W" + str(len(w)))
    print("B" + str(len(b)))
    print("S" + str(len(s)))

if __name__ == '__main__':
    
    start = time.process_time()
    #This function loads data from the text files to create a Panda Dataframe and save it as a pickle file named Sensitive_Dataset.pickle
    load_data()

    This function loads the pickle file(Created in load_data() function) into a Panda Dataframe, so that we can do further processing of data(tokenizing, lemmatizing, stopword removal etc)
    preprocess()

    This function is to view data using Dimensionality Reducation techniques
    dimension_reduction()

    print("\n\nRunning SVM...")
    svm_model_training()
    
    print("\n\nRunning KNN...")
    knn_model_training()

    print("\n\nRunning Multinomial Logistic Regression...")
    mlr_model_training()  

    print("\n\nRunning Multinomial Naive Bayes...")
    mnb_model_training()  
    
    print("\n\nRunning Random Forest...")
    rf_model_training()

    print("Running Gradient Boost...")
    gb_model_training()

    print("Running Deep Neural Network...")
    dnn_model_training()

    model_results()

    print(time.process_time() - start)