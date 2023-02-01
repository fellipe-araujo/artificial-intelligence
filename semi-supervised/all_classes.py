import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
from datetime import datetime

class Categorization():
  def load_data(self):
    # Read data from parquet
    self.data = pd.read_parquet('./labeled-data.parquet', engine='fastparquet')

    # Remove non-label from data
    for index, row in self.data.iterrows():
      if row['rotulo'] == 'Sem rotulo':
        self.data = self.data.drop(index, axis=0)
    
    # Remove data that has only one sample
    for index, row in self.data.iterrows():
      if row['rotulo'] == 'CitEdital_Cit' or row['rotulo'] == 'Susp_DivParc':
        print('here')
        self.data = self.data.drop(index, axis=0)
    
    # Set data for X and y
    self.X = self.data['texto']
    self.y = self.data['rotulo']

    # Split train and test (stratified)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42, stratify=self.y)

  def preprocessing(self):
    pass

  def train(self, pipeline):
    parameters = {
      'vect__max_df': (0.5, 0.75, 1.0),
      'vect__ngram_range': ((1, 1), (1, 2)), # unigrams or bigrams
      'clf__alpha': (0.01, 0.001, 0.0001),
    }

    scoring = {
      'accuracy',
      'f1_micro',
      'f1_macro',
      'f1_weighted'
    }

    self.grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=None, verbose=3, return_train_score=True, scoring=scoring, refit='f1_macro')

    self.grid_search.fit(self.X_train, self.y_train)

    current_date = datetime.now().strftime('./models/model_Count-NB_%d-%m-%Y_%H-%M')
    filename = current_date + '.pk'

    joblib.dump(self.grid_search, filename)

    print('Best parameters: {}'.format(self.grid_search.best_params_))
    print('Best score: {}'.format(self.grid_search.best_score_))

  def predict(self):
    self.predicted = self.grid_search.best_estimator_.predict(self.X_test)
    self.classification_report = classification_report(self.y_test, self.predicted)
    self.confusion_matrix = confusion_matrix(self.y_test, self.predicted)
    self.f1_score_macro = f1_score(self.y_test, self.predicted, average='macro')

    print(self.classification_report)
    print('F1 Score (macro): {}'.format(self.f1_score_macro))
  
  def load_model(self):
    self.grid_search = joblib.load('model_Tfid-NB_24-01-2023_18-45.pk')

if __name__ == '__main__':
  model = Categorization()

  # Load data
  model.load_data()

  # # Load model
  # model.load_model()

  # Pipeline's
  # pipeline = Pipeline([
  #   ('vect', TfidfVectorizer()),
  #   ('clf', MultinomialNB()),
  # ])

  # pipeline = Pipeline([
  #   ('vect', TfidfVectorizer()),
  #   ('clf', SGDClassifier()),
  # ])

  pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
  ])

  # pipeline = Pipeline([
  #   ('vect', CountVectorizer()),
  #   ('clf', SGDClassifier()),
  # ])

  # Train model
  model.train(pipeline)

  # Predict data
  model.predict()