import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
                #if it's a verb, return true
                if first_tag in ['VB','VBP'] or first_word == 'RT':
                    return float(1)
            except:
                # In all other cases, return false
                return float(0)

    #Given it is a transformer, return self
    def fit(self, X, y=None):
        return self

    # defining transform
    def transform(self, X):
        # applying the starting_verb function to add this feature
        X_with_starting_verb = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_with_starting_verb).dropna(inplace = True)

#custom tokenize function
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/disaster_response_cv_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    # for genre distribution graph
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # for categories distribution graph
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values

    # create visuals    
    graphs = [

        # distribution of message genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # distribution of message categories
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 40
                }
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()