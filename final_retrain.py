import sys
from keras.applications import vgg16
from keras.optimizers import SGD
from keras.layers import Input, Flatten, Dense, Dropout, MaxPooling2D, Conv2D, Activation
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

from keras.models import Model, load_model
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






def read_model(src_model):
       ## args 

       ##  list of kernels to remove.
       model = load_model(src_model)
       inp = (model.inputs[0].shape.dims[1].value,
           model.inputs[0].shape.dims[2].value,
           model.inputs[0].shape.dims[3].value)
   
       H = Input(inp)
       inp = H
   
       for i in range(len(model.layers)):
           layer = model.get_layer(index=i)
           config = layer.get_config()
 
           if isinstance(layer, MaxPooling2D):
               H = MaxPooling2D.from_config(config)(H)

           if isinstance(layer, Dropout):
               H = Dropout.from_config(config)(H)

           if isinstance(layer, Activation):
               H = Activation.from_config(config)(H)
           elif isinstance(layer, Conv2D):
               weights = layer.get_weights()
               config['trainable'] = True
               H = Conv2D(activation=config['activation'],
                       activity_regularizer=config['activity_regularizer'],
                       bias_constraint=config['bias_constraint'],
                       bias_regularizer=config['bias_regularizer'],
                       data_format=config['data_format'],
                       dilation_rate=config['dilation_rate'],
                       filters=config['filters'],
                       kernel_constraint=config['kernel_constraint'],
                       # config=config['config'],
                       # scale=config['scale'],
                       kernel_regularizer=config['kernel_regularizer'],
                       kernel_size=config['kernel_size'],
                       name=config['name'],
                       padding=config['padding'],
                       strides=config['strides'],
                       trainable=config['trainable'],
                       use_bias=config['use_bias'],
                       weights=weights
                       )(H)

           elif isinstance(layer, Flatten):
               H = Flatten()(H)

           elif isinstance(layer, Dense):
               weights = layer.get_weights()
               config['trainable'] = True
               config['name'] = config['name']+"1"
               H = Dense(units=config['units'],
                      activation=config['activation'],
                      activity_regularizer=config['activity_regularizer'],
                      bias_constraint=config['bias_constraint'],
                      bias_regularizer=config['bias_regularizer'],
                      kernel_constraint=config['kernel_constraint'],
                      kernel_regularizer=config['kernel_regularizer'],
                      kernel_initializer='glorot_uniform',
                      name=config['name'],
                      trainable=config['trainable'],
                      use_bias=config['use_bias'])(H)
 

       ## it returns the model changed 
       return Model(inp, H)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--model_input', type=str, help="input file .H5",required=True)
    parser.add_argument('--model_output', type=str, default="tmp.h5")
    args = parser.parse_args()

    output_file = args.model_output
    nb_epoch     = args.epochs
    model_input = args.model_input
    inicio  = time.time()
    lr = args.learning_rate
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
    model = read_model(model_input)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=lr,  momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

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
 
