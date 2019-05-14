import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

input_vecs = np.array(pd.read_csv("inputvector_table.txt", sep =" ", header =None))

x_train, x_test, y_train, y_test = train_test_split(input_vecs[:,2:], input_vecs[:,:2], test_size=0.33, random_state=1)

'''
    We could use normalised vectors: does this make sense?
    
    xtrain = normalize(x_train, axis=0, norm='max')
    xtest = normalize(x_test, axis=0, norm='max')
'''

#Setup autoencoder structure
input_vector = Input(shape=(16,)) #Use 16 floats vector as input
encoded_1 = Dense(500, activation='selu')(input_vector)
encoded_2 = Dense(250, activation='selu')(encoded_1)
middle_layer = Dense(2, activation='selu')(encoded_2)
decoded_1 = Dense(250, activation='selu')(middle_layer)
decoded_2 = Dense(500, activation='selu')(decoded_1)
decoded_output = Dense(16, activation='linear')(decoded_2)

'''
    Note: I have been playing around with all kinds of activation
    functions above, as well as optimizers and loss functions below.
    Please share your thoughts on which would be reasonable to use.
'''

autoencoder = Model(input_vector, decoded_output)
autoencoder.compile(optimizer='adamax', loss='mse')
autoencoder.summary()

#Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=250, #also used many different epoch values/batch sizes
                batch_size=8,
                shuffle=True,
                validation_data=(x_test, x_test))


autofitted = autoencoder.predict(x_train)

autofitted[1] - x_train[1] #compare one input with output for rough effectiveness interpretation

