from metaflow import FlowSpec, step, IncludeFile
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tabulate import tabulate
import re
import pandas as pd

class CategorizationFlow(FlowSpec):

  @step
  def start(self):
    self.next(self.load)
  
  @step
  def load(self):
    file_train = pd.read_csv('data_files/train_small.csv')
    file_test = pd.read_csv('data_files/test_small.csv')

    self.x_train = file_train['body']
    self.y_train = file_train['document_type']

    self.x_test = file_test['body']
    self.y_test = file_test['document_type']

    # remove = ('headers', 'footers', 'quotes')
    # file_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=remove)

    # self.x_train = file_train.data
    # self.y_train = file_train.target
    # self.cls = file_train.target_names

    print(self.x_train[0])
    print(self.y_train[0])
    # print(self.cls[:5])

    self.next(self.preprocessing)
  
  @step
  def preprocessing(self):
    from gpam_functions import remove_stop_words

    # Remove stopwords from train database
    data_train = list(self.x_train)
    print('X_TRAIN' + self.x_train[0])
    print('DATA_TRAIN' + data_train[0])

    i = 0
    for item in data_train:
      data_train[i] = remove_stop_words(item)
      i += 1

    self.x_train = pd.Series(data_train)

    # Remove stopwords from test database
    data_test = list(self.x_test)
    print('X_TEST' + self.x_test[0])
    print('DATA_TEST' + data_test[0])

    i = 0
    for item in data_test:
      data_test[i] = remove_stop_words(item)
      i += 1

    self.x_test = pd.Series(data_test)

    print(self.x_train[0])
    print(self.x_test[0])

    self.next(self.train)
  
  @step
  def train(self):
    pipeline = Pipeline([
      ('vect', CountVectorizer()),
      ('clf', SGDClassifier()),
    ])

    parameters = {
      'vect__max_df': ([0.5, 0.75]),
      'vect__ngram_range': ([(1, 1), (1, 2)]), # unigrams or bigrams
      'clf__alpha': ([1.0, 0.01]),
    }

    scoring = {
      'accuracy',
      'f1_micro',
      'f1_macro',
      'f1_weighted'
    }

    self.grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=None, verbose=3, return_train_score=True, scoring=scoring, refit='f1_micro')

    self.grid_search.fit(self.x_train, self.y_train)
    print('Best parameters: {}'.format(self.grid_search.best_params_))
    print('Best score: {}'.format(self.grid_search.best_score_))

    self.next(self.evaluate)
  
  @step
  def evaluate(self):
    # remove = ('headers', 'footers', 'quotes')
    # file_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42, remove=remove)

    # self.x_test = self.pipe.transform(file_test.data)
    # predicted = self.cls.predict(self.x_test)

    # print(metrics.classification_report(file_test.target, predicted, target_names=file_test.target_names))
    # print(metrics.confusion_matrix(file_test.target, predicted))

    self.predicted = self.grid_search.best_estimator_.predict(self.x_test)
    self.classification_report = metrics.classification_report(self.y_test, self.predicted)
    self.confusion_matrix = metrics.confusion_matrix(self.y_test, self.predicted)
    self.fi_score_micro = f1_score(self.y_test, self.predicted, average='micro')

    print(self.classification_report)
    print(self.confusion_matrix)
    print('F1 Score (micro): {}'.format(self.fi_score_micro))

    self.next(self.end)

  @step
  def end(self):
    pass

if __name__ == '__main__':
  CategorizationFlow()