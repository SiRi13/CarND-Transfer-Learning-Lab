import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('n_epochs', '5', "number of iterations to train (integer) DEFAULT: 5")
flags.DEFINE_integer('n_classes', '10', "number of classes for output layer (integer) DEFAULT: 10")
flags.DEFINE_integer('batch_size', '128', "batch size (integer) DEFAULT: 128")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)
    # Split data into training and validation sets.
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    n_classes = FLAGS.n_classes if FLAGS.n_classes > 0 else len(np.unique(y_train))
    input_shape = X_train.shape[1:]
    # define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # preprocess data
    X_train = np.array(X_train / 255.0 - 0.5 )
    X_val = np.array(X_val / 255.0 - 0.5 )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: train your model here
    model.fit(X_train, y_train, nb_epoch=FLAGS.n_epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
