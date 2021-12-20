##

import numpy as np
import pandas as pd
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from time import time
df = pd.read_csv('heart.csv')
target = df['target']
print(df.head())
##
df.isna().sum()
print(df.head())
##
for i in range(0, 303):
  if df['sex'][i] == 1:
   df['sex'][i] = 'Yes'
  else:
   df['sex'][i] = 'No'
for i in range(0, 303):
  if df['cp'][i] == 0:
   df['cp'][i] = 'Zero CP'
  elif df['cp'][i] == 1:
   df['cp'][i] = 'First CP'
  elif df['cp'][i] == 2:
   df['cp'][i] = 'Second CP'
  else:
   df['cp'][i] = 'Third CP'
for i in range(0, 303):
  if df['fbs'][i] == 1:
   df['fbs'][i] = 'Yes FBS'
  else:
   df['fbs'][i] = 'No FBS'
for i in range(0, 303):
  if df['restecg'][i] == 1:
   df['restecg'][i] = 'Yes ECG'
  else:
   df['restecg'][i] = 'No ECG'
for i in range(0, 303):
  if df['exang'][i] == 1:
   df['exang'][i] = 'Yes EXANG'
  else:
   df['exang'][i] = 'No EXANG'
for i in range(0, 303):
  if df['slope'][i] == 0:
   df['slope'][i] = 'Zero SLOPE'
  elif df['slope'][i] == 1:
   df['slope'][i] = 'First SLOPE'
  else:
   df['slope'][i] = 'Second SLOPE'
for i in range(0, 303):
  if df['thal'][i] == 0:
   df['thal'][i] = 'Zero THAL'
  elif df['thal'][i] == 1:
   df['thal'][i] = 'First THAL'
  elif df['thal'][i] == 2:
   df['thal'][i] = 'Second THAL'
  else:
   df['thal'][i] = 'Third THAL'
print(df.head())
df = pd.get_dummies(df, columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"], drop_first=True)
##
print(df.head())
##
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
##
X_first = X[:, 0:5]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sep = sc.fit_transform(X_first)
X = np.append(X_train_sep, X[:, 5:], axis=1)

##

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = Sequential([Dense(units=2,input_shape=(18,),activation='relu'),
                    Dense(units=5,activation='relu'),
                    keras.layers.Dropout(0.5),
                    Dense(units=10,activation='relu'),
                    keras.layers.Dropout(0.5),
                    Dense(units=2,activation='sigmoid')])
model.summary()

##
model.compile(optimizer=Adam(learning_rate=0.01),loss='sparse_categorical_crossentropy',metrics=['accuracy', 'mse'])
model.fit(
      x=X_train
    , y=y_train
    , batch_size=50
    , epochs=20
    , shuffle=True
    , verbose=0
)
predictions = model.predict(
      x=X_test
    , batch_size=20
    , verbose=0
)
rounded_predictions = np.argmax(predictions, axis=-1)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, rounded_predictions)
print(cm)
print('Test Accuracy: {}%'.format(round(accuracy_score(y_test, rounded_predictions), 4)*100))



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

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm_plot_labels = ['No Heart Disease','Heart Disease']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')