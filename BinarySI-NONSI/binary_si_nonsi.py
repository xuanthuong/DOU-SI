#  -*- coding: utf-8 -*-
"""
Binary classification - SI and NONSI
Algorithms: SVM
reference: 

Date: Jun 14, 2017
@author: Thuong Tran
@Library: scikit-learn
"""


import os, glob, random
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
import time
import matplotlib.pyplot as plt
import itertools


NEW_LINE = '\r\n'
TRAIN_SIZE = 0.8


def build_data_frame(data_dir):  
  dirs = next(os.walk(data_dir))[1] # [0]: path, [1]: folders list, [2] files list
  
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
    for f in glob.glob(os.path.join(tmp_dir, '*.txt')):
      with open(f, encoding="utf-8") as fc:
        value = [line.replace('\n', '').replace('\r', '').replace('\t', '') 
                    for line in fc.readlines()]
        value = ' '.join(value)
        rows.append({'value': value, 'class': d})
        index.append(f)

    tmp_df = DataFrame(rows, index=index)
    size = int(len(tmp_df) * TRAIN_SIZE)
    train_df, test_df = tmp_df.iloc[:size], tmp_df.iloc[size:]

    train_data = train_data.append(train_df)
    test_data = test_data.append(test_df)

    class_names.append(d)

    total_amount.append(len(os.listdir(tmp_dir)))
    train_amount.append(len(train_df))
    test_amount.append(len(test_df))
  
  tmp_arr = np.array([total_amount, train_amount, test_amount])
  print (DataFrame(tmp_arr, ['Total', 'Train', 'Test'], class_names))
  
  train_data = train_data.reindex(np.random.permutation(train_data.index))
  test_data = test_data.reindex(np.random.permutation(test_data.index))
  return train_data, test_data, class_names


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def pipeline_main():
  data_dir = '/Users/thuong/Documents/tmp_datasets/SI/SI-NonSI'
  train_data_df, test_data_df, class_names = build_data_frame(data_dir)

  class_weight = {'NonSI': 100, 'SI': 1}
  # ('tfidf_transformer',  TfidfTransformer()),
  pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', SVC(class_weight = class_weight))])

  ######### One-KFolds ##############################
  train_data, test_data = train_data_df['value'].values, test_data_df['value'].values
  train_target, test_target = train_data_df['class'].values, test_data_df['class'].values
  
  pipeline.fit(train_data, train_target)
  predictions = pipeline.predict(test_data)

  cnf_matrix = confusion_matrix(test_target, predictions)

  print('Confusion matrix with one-fold: ')
  print(cnf_matrix)
  print("One-fold: precision of NonSI: %s" % precision_score(test_target, predictions, pos_label = 'NonSI'))
  print("One-fold: precision of SI: %s" % precision_score(test_target, predictions, pos_label = 'SI'))

  precision, recall, fscore, support = score(test_target, predictions)

  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print('support: {}'.format(support))

  # # Plot non-normalized confusion matrix
  # plt.figure()
  # plot_confusion_matrix(cnf_matrix, classes=class_names,
  #                       title='Confusion matrix, without normalization')
  # # Plot normalized confusion matrix
  # plt.figure()
  # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
  #                       title='Normalized confusion matrix')
  plt.show()


def main():
  data_dir = '/Users/thuong/Documents/tmp_datasets/SI/SI-NonSI'
  train_data_df, test_data_df, class_names = build_data_frame(data_dir)

  # Feature extraction
  print ('Starting feature extraction...')
  start_time = time.time()
  count_vectorizer = CountVectorizer()
  train_data = count_vectorizer.fit_transform(train_data_df['value'].values)
  print ('Done. Feature extraction elapsed time: %s' % (time.time() - start_time))
  # print("Count Vectorize results:")
  # print(train_data.shape)
  # print(count_vectorizer.get_feature_names()[:100])
  # print("Number of features: %s" % (len(count_vectorizer.get_feature_names())))

  train_target, test_target = train_data_df['class'].values, test_data_df['class'].values

  # Training
  print ('Start training...')
  class_weight = {'NonSI': 100, 'SI': 1}
  svc_classifier = SVC(class_weight = class_weight)  
  svc_classifier.fit(train_data, train_target)
  print('training time: %s minutes' % (time.time() - start_time))

  # Testing and evaluation
  test_data = count_vectorizer.transform(test_data_df['value'].values)
  predictions = svc_classifier.predict(test_data)

  cnf_matrix = confusion_matrix(test_target, predictions)

  print('Confusion matrix with one-fold: ')
  print(cnf_matrix)
  print("One-fold: precision of NonSI: %s" % precision_score(test_target, predictions, pos_label = 'NonSI'))
  print("One-fold: precision of SI: %s" % precision_score(test_target, predictions, pos_label = 'SI'))


if __name__ == "__main__":
  main()