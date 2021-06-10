import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from helper import plot_cm
from matplotlib import pyplot

tf.config.list_physical_devices("GPU")

train_data = np.load("train_data.npy")
test_data = np.load("validation_data.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("validation_labels.npy")

# just precaution, so code will be executed on graphics card
with tf.device('/gpu:0'):

    #################################################################
    #####              MODEL INITIALIZATION                   ######
    ################################################################
    model = tf.keras.models.Sequential()
    model.add(layers.Input(shape=(98, 98, 3)))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(units=2, activation='softmax'))
    #################################################################
    #####            COMPILATION AND EXECUTION                ######
    ################################################################
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()

    X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_labels, test_size=0.2)
    hist = model.fit(X_train, y_train, batch_size=100, epochs=5, verbose=2, validation_data=(X_validation, y_validation), use_multiprocessing=True, workers=8)

    Y_test = np.argmax(test_labels, axis=1)
    y_pred = model.predict(test_data)
    Y_pred = np.argmax(y_pred, axis=1)

    plot_cm(Y_test, Y_pred)
    pyplot.show()
    print(classification_report(Y_test, Y_pred))
    
