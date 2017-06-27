#  -*- coding: utf-8 -*-
"""
Key classification
using multiclass Support Vector Machine (SVM)
reference:  

Date: Jun 05, 2017
@author: Thuong Tran
@Library: scikit-learn
"""


import os, glob, random
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
import time
import codecs
import matplotlib.pyplot as plt
import itertools


NEW_LINE = '\r\n'
TRAIN_SIZE = 0.8


def build_data_frame(data_dir):
  dirs = next(os.walk(data_dir))[1]
  
  class_names = []
  total_amount = []
  train_amount = []
  test_amount = []

  train_data = DataFrame({'value': [], 'class': []})
  test_data = DataFrame({'value': [], 'class': []})

  for d in dirs:
    tmp_dir = os.path.join(data_dir, d)
    rows = []
    index = []
    for f in os.listdir(tmp_dir):
      if f.endswith('.txt'):
        tmp_file = os.path.join(tmp_dir, f)
        with codecs.open(tmp_file, encoding="latin1") as fc:
          value = fc.read().split(NEW_LINE)
          value = '. '.join(value)
          rows.append({'value': value, 'class': d})
          index.append(tmp_file)
    tmp_df = DataFrame(rows, index=index)
    size = int(len(tmp_df) * TRAIN_SIZE)
    train_df, test_df = tmp_df.iloc[:size], tmp_df.iloc[size:]

    train_data = train_data.append(train_df)
    test_data = test_data.append(test_df)

    class_names.append(d)

    total_amount.append({d: len(os.listdir(tmp_dir))})
    train_amount.append({d: len(train_df)})
    test_amount.append({d: len(test_df)})
  
  # print ('List of classes\'s name: %s' % class_names)
  # print ('Total amount of data: %s' % total_amount)
  # print ('Total amount of train data: %s' % train_amount)
  # print ('Total amount of test data: %s' % test_amount)
  
  train_data = train_data.reindex(np.random.permutation(train_data.index))
  test_data = test_data.reindex(np.random.permutation(test_data.index))
  return train_data, test_data, class_names


def main():
  data_dir = '/Users/thuong/Documents/tmp_datasets/SI/TrainValue'
  output_model_dir = os.path.dirname(os.path.realpath(__file__))
  train_data_df, test_data_df, class_names = build_data_frame(data_dir)

  pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LinearSVC())])

  train_data, test_data = train_data_df['value'].values, test_data_df['value'].values
  train_target, test_target = train_data_df['class'].values, test_data_df['class'].values
  
  pipeline.fit(train_data, train_target)

  # save the model to disk
  filename = 'multiclass_svm_model_9keys.sav'
  joblib.dump(pipeline, os.path.join(output_model_dir, filename))


if __name__ == "__main__":
  main()