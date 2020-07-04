# Import the necessary packages
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from random import randint
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os.path
from tensorflow.keras.models import load_model

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Generate the data
train_labels = []
train_samples = []

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

for i in range(50):
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

for i in range(1000):
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

# Convert the lists to numpy array

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scalar.fit_transform(train_samples.reshape(-1, 1))

for i in scaled_train_samples:
    print(i)

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1,
          batch_size=10, epochs=30, shuffle=True, verbose=1)

test_labels = []
test_samples = []

for i in range(50):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

for i in range(50):
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

for i in range(1000):
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

# Convert the lists to numpy array

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_test_samples = scalar.fit_transform(test_samples.reshape(-1, 1))

predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
for i in predictions:
    print(i)
rounded_predictions = np.argmax(predictions, axis=-1)
for i in rounded_predictions:
    print(i)

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
cm_plot_labels = ['no_side_efffects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

if os.path.isfile('/home/anirudh/Documents/Keras_tutorials/med_trial_model.h5') is False:
    model.save('/home/anirudh/Documents/Keras_tutorials/med_trial_model.h5')

new_model = load_model('/home/anirudh/Documents/Keras_tutorials/med_trial_model.h5')
print(new_model.summary())
print(new_model.get_weights())
print(new_model.optimizer)