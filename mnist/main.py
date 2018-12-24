# pro tip:
# look up basically every method used in this program
    # import things we'll use later
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense


# cop data
# this is basically just a bunch of 28x28 arrays with a brightness value
# between 0 and 255. we need to make our network understand this data so
# what follows is a little bit of data manipulation.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape data into digestible format
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# convert to doubles
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize (reduce value to between 0 and 1) <== you want to normalize
# literally every value possible when feeding it into a network
x_train /= 255
x_test /= 255


# basically this will turn each integer in the output into a binary representation (called
# a one-hot encoding) of the input as an array. since we have numbers from 0 to 9, we have
# 10 "classes," so a number like 3 might get converted to something like:
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# to_categorical is basically a shortcut for doing this sort of classification really fast
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# throw together a network
model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="relu"))

# compile said network
model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])

# perform very fancy calculus regression wow
model.fit(x_train, y_train, batch_size=256, epochs=200, verbose=1)

# how bad did this thing fail
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: " + str(score[1]))