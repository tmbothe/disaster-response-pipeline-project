import sys
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,fbeta_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sqlalchemy import create_engine 
import pickle
import warnings
warnings.filterwarnings('always', 'ResourceWarning') 
from data.process_data import load_data


def load_data(database_filepath):
    """This function read data from a sqlite table and returns two dataframes

    Args:
        database_filepath (string): [file pattrain_classifierh to the sqllite table]

    Returns:
        [X]: [dataframe with the text data as features]
        [Y]: [dataframe with response variable; all categories]
    """
    engine = create_engine('sqlite:///database_filepath.db')
    df = pd.read_sql_table('database_filepath',con=engine)

    X = df.message
    Y = df.iloc[:,4:]
    return X , Y


def tokenize(text):
    """ This function takes text as input, then process all the transformations necessary
       to tokenize, lemmatize and returns a list with just tokens

    Args:
        text ([string]): [text that needs to be tokenized]
    """
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        
        clean_token=lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """This function builds a model to successively tokenize, transform and fit data

    Returns:
        [model]: [model]
    """
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """Print classification report for positive labels"""

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
    print("Fbeta score:", fbeta_score(y_test, y_pred, beta=2, average="weighted"))


def save_model(model, model_filepath):
    """Save the final model

    Args:
        model ([string]): model name 
        model_filepath ([string]): file path where the model is save
    """
    pickle.dump(model, open(model_filepath, "wb"))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()