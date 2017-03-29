import pickle
import tensorflow as tf
# import Keras layers you need here
from keras.models import Model
from keras.layers import Input, Dense, Flatten

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

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    n_classes = FLAGS.n_classes if FLAGS.n_classes > 0 else len(np.unique(y_train))
    in_shape = X_train.shape[1:]
    print("Shape: {}\nn_classes: {}".format(in_shape, n_classes))

    # define your model and hyperparams here
    inp = Input(shape=in_shape)
    x = Flatten()(inp)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train your model here
    model.fit(X_train, y_train, epochs=FLAGS.n_epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
