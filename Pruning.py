import numpy as np
import copy
import time
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, BatchNormalization, Add
from keras.layers import Input
from keras.models import Model
import os.path
import sys
from sklearn.metrics import log_loss, classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

class Pruning():
    __name__ = 'Pruning'

    def __init__(self, model=None, layer=1, percent = 0.1):


         self.model = model
         conv_list = self.get_conv_index()
         self.target_layer = conv_list[layer-1]
         self.changed_layer = conv_list[layer]
         weights = self.model.layers[self.target_layer].get_weights()
         self.total_drop = int(weights[0].shape[3]*percent)
         self.prunedmodel = None

    def get_conv_index(self):

        index = []
        for i in range(len(self.model.layers)):
           layer = self.model.get_layer(index=i)
           if isinstance(layer, Conv2D):
             index.append(i)
           elif isinstance(layer, Dense):  
             index.append(i)
        return index 



    def rebuild_model(self,kernel_list=None):
       ## args 

       ##  list of kernels to remove.
    
       inp = (self.model.inputs[0].shape.dims[1].value,
           self.model.inputs[0].shape.dims[2].value,
           self.model.inputs[0].shape.dims[3].value)
   
       H = Input(inp)
       inp = H
   
       for i in range(len(self.model.layers)):
           layer = self.model.get_layer(index=i)
           config = layer.get_config()
 
           if isinstance(layer, MaxPooling2D):
               H = MaxPooling2D.from_config(config)(H)

           if isinstance(layer, Dropout):
               H = Dropout.from_config(config)(H)

           if isinstance(layer, Activation):
               H = Activation.from_config(config)(H)
           elif isinstance(layer, Conv2D):
               weights = layer.get_weights()
             
               if i==self.target_layer:
                  weights[0] = np.delete(weights[0], kernel_list, axis=3)  
                  if(len(weights)==2):
                      weights[1] = np.delete(weights[1], kernel_list, 0)
               else:
                  if i==self.changed_layer:
                     weights[0] = np.delete(weights[0], kernel_list, axis=2)

               config['filters'] = weights[0].shape[3]
     
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
               if i==self.changed_layer:

                  shape = self.model.layers[i-1].input_shape
                  new_weights = np.zeros((shape[1]*shape[2]*(shape[3]-len(kernel_list)), weights[0].shape[1]))
              
                  for j in range(weights[0].shape[1]):       
                     new_weights[:,j] = np.delete(weights[0][:,j].reshape((shape[1], shape[2], shape[3])), kernel_list, 2).reshape(-1)
                  weights[0] = new_weights
                  config['units'] = weights[0].shape[1]
               H = Dense(units=config['units'],
                      activation=config['activation'],
                      activity_regularizer=config['activity_regularizer'],
                      bias_constraint=config['bias_constraint'],
                      bias_regularizer=config['bias_regularizer'],
                      kernel_constraint=config['kernel_constraint'],
                      kernel_regularizer=config['kernel_regularizer'],
                      name=config['name'],
                      trainable=config['trainable'],
                      use_bias=config['use_bias'],
                      weights=weights)(H)
 

       ## it returns the model changed 
       return Model(inp, H)


    def set_prunedmodel(self, kernel_list):

       self.prunedmodel = self.rebuild_model(kernel_list)
  


    def get_partial_model(self):

        inp = (self.prunedmodel.inputs[0].shape.dims[1].value,
           self.prunedmodel.inputs[0].shape.dims[2].value,
           self.prunedmodel.inputs[0].shape.dims[3].value)
 
        inp = Input(inp)
       
 
        i = 0
        while(i <= self.changed_layer):
            layer = self.prunedmodel.get_layer(index=i)
        
            config = layer.get_config()
          
 
            if isinstance(layer, MaxPooling2D):
                H = MaxPooling2D(pool_size=config['pool_size'],strides=config['strides'], name=config['name'])(H)

            if isinstance(layer, Dropout):
                H = Dropout.from_config(config)(H)

            if isinstance(layer, Activation):
                H = Activation.from_config(config)(H)
            elif isinstance(layer, Conv2D):
                weights = self.prunedmodel.layers[i].get_weights()
                if i==1:
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
                       )(inp)
              
                else:
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
                H = Dense(units=config['units'],
                      activation=config['activation'],
                      activity_regularizer=config['activity_regularizer'],
                      bias_constraint=config['bias_constraint'],
                      bias_regularizer=config['bias_regularizer'],
                      kernel_constraint=config['kernel_constraint'],
                      kernel_regularizer=config['kernel_regularizer'],
                      name=config['name'],
                      trainable=config['trainable'],
                      use_bias=config['use_bias'],
                      weights=weights)(H)
          
            i+=1
 
        layer = self.prunedmodel.get_layer(index=i)
        if isinstance(layer, MaxPooling2D):
           config = layer.get_config()
           H = MaxPooling2D(pool_size=config['pool_size'],strides=config['strides'], name=config['name'])(H)
           i+=1


        return Model(inp, H), (i-1)



    def sensitivity_analysis(self, X_train, Y_train, batch_size):


       l = self.model.get_layer(index=self.changed_layer)
   
       if isinstance(l, Conv2D):
          idx = -1
       else:
          idx=0

       model2 = self.rebuild_model([0])
  
       for i in range(len(self.model.layers)):
           layer = self.model.get_layer(index=i)
           if isinstance(layer, Conv2D) or isinstance(layer, Dense) :  
                weights = self.model.layers[i].get_weights()
                weights2 = model2.layers[i].get_weights()
                if weights[0].shape==weights2[0].shape:
                   model2.layers[i].set_weights(weights)
    
       weights = self.model.layers[self.target_layer].get_weights()

       ker = weights[0].shape[3] 
       best_drop = 0
       best_accuracy = 0
       kernel_result = np.zeros(ker)
       kernel_acc = np.zeros(ker)
       for i in range(ker):
        weights = self.model.layers[self.target_layer].get_weights()
        weights_2 = model2.layers[self.target_layer].get_weights() 
        weights_2[0] = np.delete(weights[0], i, axis=3)
        if len(weights)==2: 
           weights_2[1] = np.delete(weights[1], [i])
  
        model2.layers[self.target_layer].set_weights(weights_2)
        if idx<0:
                weights = self.model.layers[self.changed_layer].get_weights()
                weights_2 = model2.layers[self.changed_layer].get_weights() 
                weights_2[0] = np.delete(weights[0], i, axis=2) 
                if len(weights)==2:
                   weights_2[1] = weights[1].copy()
                model2.layers[self.changed_layer].set_weights(weights_2)
        else:
                weights = self.model.layers[self.changed_layer].get_weights()
                weights_2 = model2.layers[self.changed_layer].get_weights() 
                shape = self.model.layers[self.changed_layer-1].input_shape

                new_weights = np.zeros((shape[1]*shape[2]*(shape[3]-1), weights[0].shape[1]))

                for j in range(weights[0].shape[1]):       
                    new_weights[:,j] = np.delete(weights[0][:,j].reshape((shape[1], shape[2], shape[3])), [i], 2).reshape(-1)
                weights_2[0] = new_weights
  
                if len(weights)==2:
                    weights_2[1] = weights[1].copy()   
                model2.layers[self.changed_layer].set_weights(weights_2)
        predictions_valid = model2.predict(X_train, batch_size=batch_size, verbose=2)

        Y_pred = np.argmax(predictions_valid, axis=-1)
        Y_test = np.argmax(Y_train, axis=-1) # Convert one-hot to index
        acc = cohen_kappa_score(Y_test, Y_pred)
        loss_1 = log_loss(Y_train, predictions_valid)
        print ("loss [",i,"]= ", loss_1)
        kernel_result[i] = loss_1
        kernel_acc[i] = 1-acc

       ke = np.argsort(kernel_result)

       return ke[:self.total_drop]


    def intermediate_model(self, index):

         return  Model(inputs=self.model.input,
                                outputs=self.model.get_layer(self.model.layers[index].name).output)


    def weight_sum(self):
        weights = self.model.layers[self.target_layer].get_weights()[0]
        summation = np.sum(np.sum(np.sum(np.absolute(weights), axis=0), axis=0), axis=0)
        ke = np.argsort(summation)

        return ke[:self.total_drop]


    def APoZ(self, X_train):

       index = self.changed_layer-1
       intermediate_model = self.intermediate_model(index)
       intermediate_output = intermediate_model.predict(X_train, verbose=2)
       summation = np.count_nonzero(intermediate_output<=0, axis=0)
       summation = np.sum(np.sum(summation, axis=0), axis=0)
       summation = 1-summation/(intermediate_output.shape[1]* intermediate_output.shape[2]*intermediate_output.shape[0])
       ke = np.argsort(summation)
 
       return ke[:self.total_drop]

    

    ## retrain using different methods
    def fit_complete(self, X,Y, lr, batch_size): 
       
       if self.target_layer<= 13:
      
          print("retrain pruned model using complete model")
       
          sgd = SGD(lr=lr,  momentum=0.9, nesterov=True)
          self.prunedmodel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

          self.prunedmodel.fit(X, Y,
                   batch_size=batch_size,
                   epochs=1,
                   verbose=2 #,
                   )
          


    def fit_progressive(self, X, Y, lr, batch_size):

        modelk, index_partial = self.get_partial_model()
   
        # model to get activation in the intermediate layer
        custom_model = self.intermediate_model(index_partial)
        intermediate_output = custom_model.predict(X, verbose=2)

        sgd = SGD(lr=lr,  momentum=0.9, nesterov=True)
        modelk.compile(optimizer=sgd, loss='mse', metrics=['mse', 'mae', 'mape'])
        print("retrain model usaing progressive method")
        modelk.fit(X, intermediate_output,
                   batch_size=batch_size,
                   epochs=1,
 
                   verbose=2, #,

                   )
    
        for i in range(len(modelk.layers)):

           layer = modelk.get_layer(index=i)

           if isinstance(layer, Conv2D):
               weights = layer.get_weights()
               self.prunedmodel.layers[i].set_weights(weights)
           elif isinstance(layer, Dense):
               weights = layer.get_weights()
               self.prunedmodel.layers[i].set_weights(weights)




    def print_results(self, X, Y, batch_size):


       predictions_valid = self.prunedmodel.predict(X, batch_size=batch_size, verbose=2)

       Y_pred = np.argmax(predictions_valid, axis=-1)
       Y_test = np.argmax(Y, axis=-1) # Convert one-hot to index

       print("Accuracy = ", accuracy_score(Y_test, Y_pred))

       print(classification_report(Y_test, Y_pred))
       print("Kappa accuracy = ", cohen_kappa_score(Y_test, Y_pred))
       print( confusion_matrix(Y_test, Y_pred))


    def save_model(self, output):
       self.prunedmodel.save(output)     


  

      


