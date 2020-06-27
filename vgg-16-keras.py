import sys
from keras.applications import vgg16
from keras.optimizers import SGD
from keras.layers import Input, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from keras.models import Model
import numpy as np
import time
import argparse
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import keras.backend as K
import keras



class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))


def vgg16_model(shape1, num_classes):

  img_input = Input(shape=shape1)

  model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=img_input, pooling=None, classes=num_classes)

  
  x = model.output

  x = Flatten(name='flatten_a')(x)
  x = Dense(4096, activation='relu', name='fc1_a',  kernel_initializer='glorot_uniform')(x)
  x = Dropout(0.5)(x)
  x = Dense(4096, activation='relu', name='fc2_a',  kernel_initializer='glorot_uniform')(x)
  x = Dropout(0.5)(x)
  x = Dense(num_classes, activation='softmax', name='predictionsa')(x)
  model = Model(img_input, x)
  
    # Learning rate is changed to 0.001
  sgd = SGD(lr=1e-5,  momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--model_output', type=str, default="tmp.h5", required=True)
    args = parser.parse_args()

    output_file = args.model_output
    nb_epoch     = args.epochs

    inicio  = time.time()

    ## Parameters to be set
    img_rows, img_cols = 32, 32 # Resolution of inputs
    channel      = 3

    num_classes  = 10
    batch_size   = 8

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


    X_train, X_test = X_train.astype('float32')/255, X_test.astype('float32')/255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Load our model
    model = vgg16_model((img_rows, img_cols, channel), num_classes=num_classes)



    mcp = ModelCheckpoint(output_file, monitor="loss", mode="min",
                             save_best_only=True, save_weights_only=False)
    lr = PrintLearningRate()

    # Start Fine-tuning
    print("Finetune network")
    model.fit(X_train, y_train,
                   batch_size=batch_size,
                   epochs=nb_epoch,
                   shuffle=False,
                   verbose=2,          
                   callbacks=[mcp, lr]
 
                   )

    model.load_weights(output_file)

       
    # Show layer's name
    for i in range(len(model.layers)):
      print (str(i) , model.layers[i].name)
   

    predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=2)

    Y_pred = np.argmax(predictions_valid, axis=-1)
    Y_test = np.argmax(y_test, axis=-1) # Convert one-hot to index

    print("Accuracy = ", accuracy_score(Y_test, Y_pred))

    print(classification_report(Y_test, Y_pred))
    print("Kappa accuracy = ", cohen_kappa_score(Y_test, Y_pred))
    print( confusion_matrix(Y_test, Y_pred))
    fim = time.time()
    total = fim-inicio
    print( "Execution time - ", fim-inicio," sec")
 
