import pandas as pd
from pandas import DataFrame
import functools, itertools
import glob, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


SIData_dir = '/Users/thuong/Documents/SI_Data'
df_arr = []
for f in glob.glob(os.path.join(SIData_dir, '*.csv')):
    df_arr.append(pd.read_csv(f, encoding='latin1'))
df = functools.reduce(lambda left, right: pd.merge(left, right, on=['BKG_NO']), df_arr)

df.replace({'#': ' '}, regex=True, inplace=True)
df.replace({'\$': ' '}, regex=True, inplace=True)

from sklearn.feature_extraction import text
stop_words = set(text.ENGLISH_STOP_WORDS)

def text_cleaning(text):
    words = []
    words.extend(w for w in str(text).split() 
                    if w.isalpha() and len(w) != 1 and w.lower() != 'nan' 
                                    and w.lower() not in stop_words)
    return ' '.join(words)


df_data = DataFrame(columns=['Value', 'Class'])
class_names = []
for col in df:
    if col != 'BKG_NO':
        df[col] = df[col].apply(text_cleaning)
        tmp_df = DataFrame(columns=['Value', 'Class'])
        tmp_df['Value'] = df[col]
        tmp_df['Class'] = col
        df_data = df_data.append(tmp_df)
        class_names.append(col)
    else:
        df[col] = df[col]
df_data = df_data[df_data['Value'] != ""]
df_data = df_data.reset_index(drop=True)


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


df_data = df_data.reindex(np.random.permutation(df_data.index))

TRAIN_SIZE = 0.8
size = int(len(df_data) * TRAIN_SIZE)
df_train, df_test = df_data.iloc[:size], df_data.iloc[size:]

total_amount = [len(df_data[df_data['Class'] == c]) for c in class_names]
train_amount = [len(df_train[df_train['Class'] == c]) for c in class_names]
test_amount = [len(df_test[df_test['Class'] == c]) for c in class_names]
tmp_arr = np.array([total_amount, train_amount, test_amount])
print(DataFrame(tmp_arr, ['Total', 'Train', 'Test'], class_names))


pipeline = Pipeline([
                    ('vectorizer', CountVectorizer()),
                    ('tfidf_transformer',  TfidfTransformer()),
                    ('classifier', LinearSVC())])

train_data, test_data = df_train['Value'].values, df_test['Value'].values
train_target, test_target = df_train['Class'].values, df_test['Class'].values

pipeline.fit(train_data, train_target)
predictions = pipeline.predict(test_data)

cnf_matrix = confusion_matrix(test_target, predictions)
print("Score with one-fold: %s" % precision_score(test_target, predictions, average = 'weighted'))
print("Score with one-fold: %s" % precision_score(test_target, predictions, average = None))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
plt.show()