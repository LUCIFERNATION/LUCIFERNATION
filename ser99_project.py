# -*- coding: utf-8 -*-
"""SER99_PROJECT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-DrxGZv4DOXD9PP7U6dOEbuCidvyp-_e

# **UPLOADING** THE **DATASET** WITH **GOOGLE** **DRIVE**
"""

from google.colab import drive
drive.mount('/content/drive')

import os     #OS = Operation System
Root = "/content/drive/MyDrive/TP-Leclass/RAVDESS"
os.chdir(Root)

ls

"""# **EDA — Exploratory Data Analysis - does this for Machine Learning enthusiast**"""

# Commented out IPython magic to ensure Python compatibility.
# IMPORT NECESSARY LIBRARIES
import librosa    # Audio
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import numpy as np   # ARray 
import tensorflow as tf  # Deep learnin
import tensorflow.keras   # Deep Learning
from matplotlib.pyplot import specgram
import pandas as pd
import IPython.display as ipd  # To play sound in the notebook
import os # interface with underlying OS that python is running on
import sys
import soundfile as sf
from sklearn.model_selection import StratifiedShuffleSplit    # Machine learning library
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.validation import column_or_1d
import warnings # ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""**EXEMPLE** : **ACTOR** **NUMBER** **17**  **WHERE X IS THE DATA & SR IS THE SAMPLE RATE**

***NEUTRAL MALE***
"""

# LOAD IN FILE FOR EXEMPLE : ACTOR 17
x, sr = librosa.load('/content/drive/MyDrive/TP-Leclass/RAVDESS/Actor_17/03-01-01-01-01-01-17.wav')
# PLAY AUDIO FILE
sf.write('/content/drive/MyDrive/TP-Leclass/RAVDESS/Actor_17/03-01-01-01-01-01-17.wav', x, sr)
Audio(data=x, rate=sr)

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveplot(x, sr=sr)
plt.title('Waveplot - Actor 17 is Neutral -')

"""**Mel Spectogram**"""

# CREATE LOG MEL SPECTROGRAM FOR ACTOR 17 FOR EXEMPLE
spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128,fmax=8000) 
spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time');
plt.title('Actor 17 is Neutral - Mel Spectrogram ')
plt.colorbar(format='%+2.0f dB');

"""**Function extract_feature to extract the mfcc, chroma, and mel features from a sound file**"""

#Emotions in the RAVDESS dataset - Dictionnary
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

# Filename identifiers   # Indices

   # Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
   # Vocal channel (01 = speech, 02 = song).
   # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
   # Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
   # Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
   # Repetition (01 = 1st repetition, 02 = 2nd repetition).
   # Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

# CREATE DIRECTORY OF AUDIO FILES 
audio = "/content/drive/MyDrive/TP-Leclass/RAVDESS/"
actor_folders = os.listdir(audio) #list files in audio directory
actor_folders.sort() 
# CREATE FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
emotion = []
gender = []
actor = []
file_path = []
for i in actor_folders:          # iterate over all the actors in the RAVDESS DATASET FOR EACH ACTOR
    filename = os.listdir(audio + i) ## OS.LISTDIR ===> Returns a list containing the names of the files in the directory : ACTOR(i) when 01 < i < 24
    for f in filename:        # go through file NAMES WHICH INCLUDE EMOTIONS THEMSELVES in Actor folder
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        actor.append(int(part[6]))  #(7 items) ['03', '01', '08', '02', '02', ...]
        bg = int(part[6])
        if bg%2 == 0:
            bg = "female"
        else:
            bg = "male"
        gender.append(bg)
        file_path.append(audio + i + '/' + f)  # Getting all paths for every Actors

           #### PUT EXTRACTED LABELS WITH FILEPATH INTO DATAFRAME
audio_df = pd.DataFrame(emotion)
audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fearful', 7:'disgust', 8:'surprised'})
audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
audio_df.columns = ['gender','emotion','actor']
audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)
audio_df = audio_df.iloc[:-1 , :]
audio_df.shape

# ENSURE GENDER,EMOTION, AND ACTOR COLUMN VALUES ARE CORRECT
pd.set_option('display.max_colwidth', None)

audio_df.sample(10)   #échantillon

# LOOK AT DISTRIBUTION OF CLASSES
audio_df.emotion.value_counts().plot(kind='bar')

"""**Emotions**

"""

# ITERATE OVER ALL AUDIO FILES AND EXTRACT LOG MEL SPECTROGRAM MEAN VALUES INTO DF FOR MODELING 
df = pd.DataFrame(columns=['mel_spectrogram'])
counter=0
for index,path in enumerate(audio_df.path):    #enumerate ( , ) ==> ('index', path)*audio_df times
  X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)     # Load all paths iteratively as a floating point time series 
                                          # high-quality mode (‘kaiser_best’)
#get the mel-scaled spectrogram (transform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes)    
  spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)     
  db_spec = librosa.power_to_db(spectrogram)    #temporally average spectrogram  (Convert a power spectrogram (amplitude squared) to decibel (dB) units)  
  log_spectrogram = np.mean(db_spec, axis = 0)      # Average values into log_spectrogram               
  df.loc[counter] = [log_spectrogram]    
  counter=counter+1

print(len(df),'rows')
df.head()

"""**LOAD THE DATA & EXTRACT FEATURES FOR EACH FILE**

# **Data** **preprocessing**

**I used pd.concat to turn the array into a list and join with my previous DataFrame audio_df, and dropped the necessary columns to give us the final DataFrame.**
"""

# TURN ARRAY INTO LIST AND JOIN WITH AUDIO_DF TO GET CORRESPONDING EMOTION LABELS
df_combined = pd.concat([audio_df,pd.DataFrame(df['mel_spectrogram'].values.tolist())],axis=1)
df_combined = df_combined.fillna(0)
# DROP PATH COLUMN FOR MODELING
df_combined.drop(columns='path',inplace=True)
# CHECK TOP 5 ROWS
df_combined.head()

"""# **SPLITTING THE DATA INTO A TRAINING** **SET** & **A TEST SET**"""

# TRAIN TEST SPLIT DATA
train , test = train_test_split(df_combined, test_size=0.2, random_state=0, stratify=df_combined[['emotion','gender','actor']])
X_train = train.iloc[:, 3:]
y_train = train.iloc[:,:2].drop(columns=['gender'])
X_test = test.iloc[:,3:]
y_test = test.iloc[:,:2].drop(columns=['gender'])
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Stand DATA
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train.head()

"""# **SVM**"""

#Initialize the SVM Classifier
from sklearn.svm import SVC
SVM_Classifier = SVC(C=1, kernel = 'rbf', probability = True , decision_function_shape='ovo' , random_state = 42)
SVM_Classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = SVM_Classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)

#Calculate the accuracy of our model
accuracy_SVM = accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Accuracy Score for SVM : {:.2f}%".format(accuracy_SVM*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = SVM_Classifier, X = X_train, y = y_train, cv = 3)

print("3Folds Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print(classification_report(y_test, y_pred))

"""# **Decision Tree**"""

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_DTC = DecisionTreeClassifier(criterion = 'gini', max_leaf_nodes = 24 , splitter = 'best' ,max_features = 'log2', random_state = 42)
classifier_DTC.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_DTC.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm_Tree = confusion_matrix(y_test, y_pred)
print(cm_Tree)
# Accuracy Score of the Decision Tree MODEL
Accuracy_DTC = accuracy_score(y_test, y_pred)
print(Accuracy_DTC)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_DTC, X = X_train, y = y_train, cv = 3)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print(classification_report(y_test, y_pred))

"""# **CNN** **MODEL**

***TURNING DATA INTO ARRAYS FOR KERAS + LABEL ENCODING***
"""

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# ONE HOT ENCODE THE TARGET
# CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

print(y_test[0:10])

print(lb.classes_)

#X_train = np.expand_dims(X_train, axis=2)
#X_test = np.expand_dims(X_test, axis=2)
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

# RESHAPE DATA TO INCLUDE 3D TENSOR 
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#nsamples, nx, ny = X_train.shape
#X_train = X_train.reshape((nsamples,nx*ny))
#print(X_train)
#nsamplese, nxe, nye = X_test.shape
#X_test = X_test.reshape((nsamplese,nxe*nye))

"""***BUILDING MY OWN CNN ARCHITECTURE***"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
#BUILDING MY OWN CNN ARCHITECTURE
model = tf.keras.Sequential()
model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1)))
model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Dropout(0.4))
model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(8, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

"""**FIT MODEL AND USE CHECKPOINT TO SAVE BEST CNN MODEL**"""

import tensorflow.keras as keras

# FIT MODEL AND USE CHECKPOINT TO SAVE BEST MODEL
checkpoint = ModelCheckpoint("99_cnn_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_freq=1, save_weights_only=True)

model_history=model.fit(np.array(X_train), y_train,batch_size=32, epochs=188, validation_data=(X_test, y_test),callbacks=[checkpoint])

# PLOT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""**PRINTING THE LOSS & ACCURACY PERCENTAGE ON THE TEST SET**

"""

print("Loss of the model is - " , model.evaluate(X_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(X_test,y_test)[1]*100 , "%")

# PREDICTIONS
predictions = model.predict(X_test)
predictions=predictions.argmax(axis=1)
predictions = predictions.astype(int).flatten()
predictions = (lb.inverse_transform((predictions)))
predictions = pd.DataFrame({'Predicted Values': predictions})

# ACTUAL LABELS
actual=y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform((actual)))
actual = pd.DataFrame({'Actual Values': actual})

# COMBINE BOTH 
finaldf = actual.join(predictions)
finaldf[68:88]

print(classification_report(actual, predictions, target_names = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']))

"""# **CNN TRANSFER LEARNING : vgg16, vgg19, Xception, ResNet50, MobileNetV2, DenseNet121, EfficientNetB5.**"""

!pip install keras-tqdm
!pip install pycm

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import time
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.efficientnet import EfficientNetB5


# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from keras_tqdm import TQDMNotebookCallback
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras.layers import Dense,GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from pycm import *
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from numpy import array
from sklearn.model_selection import KFold
from keras.preprocessing.image import load_img
from tqdm import tqdm_notebook as tqdm
from keras_tqdm import TQDMNotebookCallback
from keras_tqdm import TQDMCallback
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import MaxPooling2D
import keras
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Flatten
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
def plot_acc_loss(history, PLOT_NAME, Folder_Training_name):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
 
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
def getPrediction(l):
  res = []
  i=0
  while i<len(l):
    if(l[i][0]==True):
      res.append(1)
    else:
      res.append(0)
    i+=1
  return res
#---------------------------------------------------------------------- Main
all_deep_models = [VGG16, VGG19, Xception, DenseNet121, MobileNetV2, ResNet50, EfficientNetB5]
all_model_name_txt = ["VGG16", "VGG19", "Xception", "DenseNet121", "MobileNetV2", "ResNet50", "EfficientNetB5"]



CLASSES = 2
WIDTH = 640
HEIGHT = 480
BATCH_SIZE = 10
EPOCHS = 5
learning_rate = 0.01
all_Kfolds = ["_kfold1", "_kfold2", "_kfold3"]

all_folder_training_names = [
                             "/TP-Leclass/SER_transfer_lear/ourdataset/train"
                             ]

for Folder_Training_name in all_folder_training_names:
  print ("---------------------------------------------------------------------------")
  print ("- Training Folder Name: "+str(Folder_Training_name))
  print ("---------------------------------------------------------------------------")
  i=0
  for deep_model in all_deep_models:
    model_name = deep_model
    model_name_txt = str(all_model_name_txt[i])
    print ("- Deep model: "+str(model_name_txt))
    for KFOLD_ in all_Kfolds:
      
      print ("-KFold: "+str(KFOLD_))
      
      
      TRAIN_DIR = "/content/drive/MyDrive"+str(Folder_Training_name)+"/train"+str(KFOLD_)+"/"
      TEST_DIR = "/content/drive/MyDrive"+str(Folder_Training_name)+"/test"+str(KFOLD_)+"/"
      FOLD_NAME = str(model_name_txt)+str(KFOLD_)+".txt"
      Train_TimeFOLD_NAME = "TrainingTime_"+str(model_name_txt)+str(KFOLD_)+".txt"
      PLOT_NAME = model_name_txt+"/"+str(model_name_txt)+str(KFOLD_)+".png"

      # setup model
      base_model = model_name(weights='imagenet', include_top=False, input_tensor=Input(shape=(480, 640, 3)))
      print ("# Base model architecture before update")
      base_model.summary()

      x=base_model.output

      x=GlobalAveragePooling2D()(x)
      x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
      x=Dense(512,activation='relu')(x) #dense layer 2
      x=Dense(128,activation='relu')(x) #dense layer 3
        #Batch normalization is a technique for training very deep neural networks 
        #that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process 
        #and dramatically reducing the number of training epochs required to train deep networks.
      x = BatchNormalization()(x)
      preds=Dense(1, activation='sigmoid')(x) #final layer with softmax activation
      model = Model(inputs=base_model.input, outputs=preds)
      # Say not to train first layer (ResNet) model as it is already trained
      model.layers[0].trainable = False
      adam = Adam(learning_rate = learning_rate)
      model.compile(optimizer = adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

      print ("# Final model architecture after update")
      model.summary()
    #for bi class: loss = 'categorical_crossentropy'
    #for muti class: loss='categorical_crossentropy'


      # data prep
      train_datagen = ImageDataGenerator()
      validation_datagen = ImageDataGenerator()
      print ("-- Load Training data: ")
      train_generator = train_datagen.flow_from_directory(TRAIN_DIR,target_size=(HEIGHT, WIDTH),batch_size=BATCH_SIZE,class_mode='categorical')
      print ("-- Load Test data: ")
      validation_generator = validation_datagen.flow_from_directory(
          TEST_DIR,
          target_size=(HEIGHT, WIDTH),
          batch_size=BATCH_SIZE,
          shuffle=False,
          class_mode='categorical')

      print ("------ validation_generator.class_indices ----------")
      print (validation_generator.class_indices)
      print ("----------------------------------------------------")
      STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
      STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size


      STEPS_PER_EPOCH = STEP_SIZE_TRAIN
      VALIDATION_STEPS = STEP_SIZE_VALID

      #STEPS_PER_EPOCH = 5
      #VALIDATION_STEPS = 5
      print ("- EPOCHS: "+str(EPOCHS))
      print ("- BATCH_SIZE: "+str(BATCH_SIZE))
      #print ("- STEPS_PER_EPOCH: "+str(STEPS_PER_EPOCH))
      #print ("- VALIDATION_STEPS: "+str(VALIDATION_STEPS))
      #1 Epoch = 1 Forward pass + 1 Backward pass for ALL training samples.
      #Batch Size = Number of training samples in 1 Forward/1 Backward pass. 
      # (With increase in Batch size, required memory space increases.)
      #Number of iterations = Number of passes i.e. 1 Pass = 1 Forward pass + 1 Backward pass 
      # (Forward pass and Backward pass are not counted differently.)

      MODEL_FILE = "/content/drive/MyDrive"+str(Folder_Training_name)+"/output/"+str(model_name_txt)+"/"+KFOLD_+str(model_name_txt)+".model"
      start_time = time.time()
      
      checkpoint = ModelCheckpoint(MODEL_FILE,
                                  monitor='val_accuracy',
                                  verbose=1,
                                  mode='max',
                                  save_best_only=True,
                                  save_weights_only=False,
                                  save_freq=1)
      
      history = model.fit(
          train_generator,
          epochs = EPOCHS,
          steps_per_epoch = 50,
          validation_steps = VALIDATION_STEPS,
          validation_data = validation_generator,callbacks=[checkpoint])
      #callbacks=[checkpoint]


      end_time = time.time()
      duration = end_time - start_time
      print("Total training time = "+str(duration))
      tt = "/content/drive/MyDrive"+str(Folder_Training_name)+"/output/"+str(model_name_txt)+"/"+Train_TimeFOLD_NAME
      f = open(tt, "w")
      f.write("\n"+str(duration))
      f.close()
      model.save(MODEL_FILE)
      plot_acc_loss(history, PLOT_NAME, Folder_Training_name)


      # evaluate loaded model on test data
      # load model
      model = load_model(MODEL_FILE)
      # summarize model.
      model.summary()
      # evaluate the model

      training_score = model.evaluate_generator(train_generator,  verbose=1)
      print("- Training accuracy: "+str(training_score[1]*100))
      validation_score = model.evaluate_generator(validation_generator,  verbose=1)
      print("- Validation accuracy: "+str(validation_score[1]*100))

      #Confution Matrix and Classification Report
      probabilities = model.predict_generator(validation_generator)
      y_pred = probabilities > 0.5
      y_true = validation_generator.classes
      print('Confusion Matrix')
      print(confusion_matrix(y_true, y_pred))
      print('Classification Report')
      target_names = {'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}
      print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
      print ("--- y_true")
      print (y_true)
      print ("--- y_pred")
      y_pred = np.array(getPrediction(y_pred))

      print (y_pred)
      results = "/content/drive/MyDrive"+str(Folder_Training_name)+"/output/"+str(model_name_txt)+"/"+FOLD_NAME
      f = open(results, "w")
      cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred) # Create CM From Data
      f.write("{}\n".format(cm))
      f.close()
    i+=1

"""# **FLASK**

"""

!pip install qiskit ipywidgets
!pip install flask-ngrok
!pip install flask==2.0.2

# flask_ngrok_example.py
from flask import Flask
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

@app.route("/")
def hello():
    return "<h1> Hello! IT'S LUCIFER99 ! </h1>"

if __name__ == '__main__':
    app.run()  # If address is in use, may need to terminate other sessions:
               # Runtime > Manage Sessions > Terminate Other Sessions

"""# **Fine Tuning using GridSearch Cross Validation**"""

df = pd.read_csv('/content/drive/MyDrive/TP-Leclass/audio.csv')
df.head()

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
           }
    },     
}

from sklearn.model_selection import GridSearchCV
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False)
    clf.fit(digits.data, digits.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df

from keras.utils import np_utils, to_categorical
# TRAIN TEST SPLIT DATA
train,test = train_test_split(df_combined, test_size=0.2, random_state=0,
                               stratify=df_combined[['gender','actor']])

X_train = train.iloc[:, 3:]
y_train = train.iloc[:,:2].drop(columns=['gender'])
print(X_train.shape)

X_test = test.iloc[:,3:]
y_test = test.iloc[:,:2].drop(columns=['gender'])
print(X_test.shape)
# NORMALIZE DATA
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
# TURN DATA INTO ARRAYS FOR KERAS
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
# RESHAPE TO INCLUDE 3D TENSOR
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# GRID SEARCH PARAMETERS TO FIND BEST VALUES
classifier = KerasClassifier(build_fn = make_classifier)
params = {
    'batch_size': [30, 32, 34],
    'nb_epoch': [25, 50, 75],
    'optimizer':['adam','SGD']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring='accuracy',
                           cv=3)

grid_search = grid_search.fit(X_train,y_trainHot)
grid_search.best_params_
grid_search.best_score_

"""# **Personal Approach**#

 ***WE CAN NOW build system that can recognize emotion in real time and then calculate degree of affection such as love, truthfulness, and friendship of the person you are talking to.***
"""