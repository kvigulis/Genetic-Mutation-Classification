# Handle table-like data and matrices
import numpy as np
import pandas as pd

import keras
import keras.utils
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from sklearn.model_selection import train_test_split , StratifiedKFold

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#submission_df = pd.read_csv("C:/Users/Karlis/PycharmProjects/PandasTest/GeneticMutationsDataSet/submissionFile.csv")
train_variants_df = pd.read_csv("C:/Users/Karlis/PycharmProjects/PandasTest/GeneticMutationsDataSet/training_variants.csv")
test_variants_df = pd.read_csv("C:/Users/Karlis/PycharmProjects/PandasTest/GeneticMutationsDataSet/test_variants.csv")
train_text_df = pd.read_csv("C:/Users/Karlis/PycharmProjects/PandasTest/GeneticMutationsDataSet/training_text.csv", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("C:/Users/Karlis/PycharmProjects/PandasTest/GeneticMutationsDataSet/test_text.csv", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

# Merge Variants and Text Data:
full_training = train_variants_df.merge(train_text_df, how="inner", left_on="ID", right_on="ID")
full_test = test_variants_df.merge(test_text_df, how="inner", left_on="ID", right_on="ID")

# Append Test data to Training data
full = train_variants_df.append( test_variants_df , ignore_index = True )
train = full[ :3321 ]

genes = pd.get_dummies( full.Gene , prefix='Gene' )
variations = pd.get_dummies( full.Variation , prefix='Variation' )

full_X = pd.concat([genes] , axis=1 )

# Create all datasets that are necessary to train, validate and test models
train_X = full_X[ 0:3321 ]
train_y = pd.get_dummies( train_variants_df.Class , prefix='class', prefix_sep='' )
test_X = full_X[ 3321: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_X , train_y , train_size = .9 )

print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
train_X = train_X.as_matrix()
valid_X = valid_X.as_matrix()
train_y = train_y.as_matrix()
valid_y = valid_y.as_matrix()
test_X = test_X.as_matrix()

layer_1 = 64
layer_2 = 18
dropout_1 = 0.5
dropout_2 = 0.5

model = Sequential()
model.add(Dense(layer_1, activation='relu', input_dim=1507)) #1507 and 10116
#model.add(Dropout(dropout_1))
#model.add(Dense(layer_2, activation='relu')) #1507 and 10116
#model.add(Dropout(dropout_2))
model.add(Dense(9, activation='softmax'))

adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adagrad,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_X , train_y, validation_data=(valid_X,valid_y), epochs=30, batch_size=10, initial_epoch=0)

#print (model.evaluate( train_X , train_y ))
print ("\n\nEvaluation:\n\nTraining score:", model.evaluate(train_X, train_y))
print("\nValidation score:", model.evaluate(valid_X, valid_y))
print("Layer 1: ", layer_1, "; Later 2: ", layer_2)
print("Dropout 1: ", dropout_1, "; Dropout 2: ", dropout_2)

test_Y = model.predict( test_X )
test_Y = pd.DataFrame(test_Y)
test_Y.columns = ['class1','class2','class3','class4','class5','class6','class7','class8','class9']
test_Y = pd.concat( [ test_variants_df.ID, test_Y ] , axis=1 )
test_Y.to_csv( "C:/Users/Karlis/PycharmProjects/PandasTest/GeneticMutationsDataSet/NN1024_64.csv" , index = False )


# Class frequency plot
sns.set()
fig, ax = plt.subplots(nrows=2, figsize=(12,18))
ax[0].plot(history.history['acc'], 'm')
ax[0].plot(history.history['val_acc'], 'r')
ax[0].set_title("Model Accuracy")
ax[0].set_xlabel("Epochs")
ax[0].legend(['train', 'test'], loc='upper left')

ax[1].plot(history.history['loss'], 'k')
ax[1].plot(history.history['val_loss'] , 'b')
ax[1].set_title("Model Loss")
ax[1].set_xlabel("Epochs")
ax[1].legend(['train', 'test' ], loc='upper left')
plt.show()

