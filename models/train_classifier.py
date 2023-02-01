# import libraries
import sys
import os
import re
import pickle
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, precision_recall_curve, precision_recall_fscore_support, multilabel_confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def load_data(database_filepath):
    """This function reads the table from sqlite database, performs basic cleaning
     and separates messages and categories

    Args:
        database_filepath (str): file path for database (.db file)

    Returns:
        X (pd.Series): This is a pandas series containing messages.
        These are the inputs for the classifier.
        Y (pd.DataFrame): a data frame containing one column for each output class (category)
        Y.columns (list): target labels (category names)  
    """

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)

    # Value 2 in the related field is an error. Replacing 2 with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. Since there is no information, majority class (1) is considered.
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    # Remove child alone as it has all zeros only
    df = df.drop(['child_alone'], axis=1)

    # separate feature and target into X and Y
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)

    return X, Y, Y.columns


def tokenize(text):
    """This function cleans text and lemmatizes it to return cleaned tokens

    Args:
        text (str): message

    Returns:
        tokens (list): cleaned tokens
    """
    # removing punctuations, lowering the case and stripping leading and trailing space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize initialization
    lemmatizer = WordNetLemmatizer()

    # lemmatize and store clean tokens
    tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    # return cleaned tokens
    return tokens

# Building a custom transformer to extract the starting verb of a sentence


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    This class extract the starting verb of the message,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):

        # sentence tokenizing the text
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:
            # tokenize each sentence and tag parts of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))

            try:
                # extract the first word and its tagged part of speech
                first_word, first_tag = pos_tags[0]
                # if it's a verb, return true
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return float(1)
            except:
                # In all other cases, return false
                return float(0)

    # Given it is a transformer, return self
    def fit(self, X, y=None):
        return self

    # defining transform
    def transform(self, X):
        # applying the starting_verb function to add this feature
        X_with_starting_verb = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_with_starting_verb).dropna(inplace=True)


def build_model():
    """This function adds features to training data and builds a classifier

    Returns:
        cv (sklearn.pipeline.Pipeline): a pipeline with features and classifier along with parameters
        ready for grid search and training.
    """
    # Pipeline with count vecotrizer, tdidf transformer, added feature - StartingVerbExtractor and an ensemble classifier with best parameters from cross validation
    pipeline_starting_verb_ext = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(
            AdaBoostClassifier()))
    ])

    # parameters for adaboost classifier
    parameters = {
        'clf__estimator__n_estimators': [10, 100],
        'clf__estimator__learning_rate': [0.1, 1]
    }

    # passing new pipeline and parameters for grid search
    cv = GridSearchCV(pipeline_starting_verb_ext, param_grid = parameters, scoring = 'f1_micro', n_jobs = -1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """This function performs prediction and reports precision, recall and f1 score metrics

    Args:
        model (sklearn.pipeline.Pipeline): a trained model pipeline
        X_test (pd.Series): messages from test data
        Y_test (pd.DataFrame): categories from test data
        category_names (list): target names or category names

    Returns:
        NoneType: prints precision, recall and f1 score for each category
    """
    # predicting with best parameters
    y_pred = model.predict(X_test)

    # printing precision, recall, f1 score
    return print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """This function saves the model in pickle format in the given model_filepath

    Args:
        model (sklearn.pipeline.Pipeline): a trained model pipeline
        model_filepath (str): file path to save the model pickle

    Returns: None

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
