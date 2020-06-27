from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation

from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D 
from keras import backend
import time
import sys
import json
import numpy as np
from sklearn.metrics import log_loss, classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
import keras.backend as K
import argparse
import os
from Pruning import Pruning
import keras
	


if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10
    inicio  = time.time()


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input', '-mi', type=str, required=True)
    parser.add_argument('--p', type=float, default=0.05)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--layer_number', '-ln', type=int, help='layer index = 1,...,13', required=True)
    parser.add_argument('--model_output', '-mo', type=str, default="2", required=True)
    parser.add_argument('--retrain', '-r', type=str, default="complete") 
    parser.add_argument('--criterion', '-c', type=str, default="sensitivity")
    args = parser.parse_args()


    p = args.p
    model_input = args.model_input
    output = args.model_output
    target_layer = args.layer_number
    criterion = args.criterion
    retrain = args.retrain

    ## Parameters to be set
    img_rows, img_cols = 32,32 # Resolution of inputs
    channel      = 3
    num_classes  = 10
    batch_size   = 8
   

    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    X_train, X_test = X_train.astype('float32')/255, X_test.astype('float32')/255
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    model = load_model(model_input)

    Pruning_method = Pruning(model=model, layer=target_layer, percent=p)



    ## step one: Choose kernels according to criterion

    
    if criterion=="sensitivity":
       ## perform sensitivity analysis
       kernels_to_prune = Pruning_method.sensitivity_analysis(X_train, Y_train, batch_size)
    elif criterion=="weight_sum":
         ## weight sum
         kernels_to_prune = Pruning_method.weight_sum()
    else:
         ## APoZ
         kernels_to_prune = Pruning_method.APoZ(X_train)
    

 
    ## Step 2: Prune kernels from original model
    #kernels_to_prune = [0,5,7]
    Pruning_method.set_prunedmodel(kernels_to_prune)




    ## Step 3: retraining to recover generalization
    ## For progressive training, we used different learning rate in each pruned layer
    ## Pruned layer   lr
    ## block1_conv1   1e-5
    ## block1_conv2   1e-5
    ## block2_conv1   1e-6
    ## block2_conv2   1e-6
    ## block3_conv1   1e-7
    ## block3_conv2   1e-7
    ## block3_conv3   1e-7
    ## block4_conv1   1e-8
    ## block4_conv2   1e-8
    ## block4_conv3   1e-8
    ## block5_conv1   1e-8
    ## block5_conv2   1e-8
    ## block5_conv3   1e-8

    




    if retrain=="complete":
       Pruning_method.fit_complete(X=X_train, Y=Y_train, lr=1e-5, batch_size=batch_size)
    else:
       lr = [1e-5, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
       Pruning_method.fit_progressive(X=X_train, Y=Y_train, lr=lr[target_layer-1], batch_size=batch_size)

    
  

    ## print accuracy and cohen kappa
    Pruning_method.print_results(X_test, Y_test, batch_size)


    ## save pruned model
    Pruning_method.save_model(output)
   
    fim = time.time()
    total = fim-inicio
    print( "Execution time - ", fim-inicio," sec")
    sec = total%60
    b =  total / 60
    hour = b / 60
    min = b % 60
    print (int(hour),":", int(min), ":", int(sec))
