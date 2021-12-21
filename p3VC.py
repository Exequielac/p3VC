# -*- coding: utf-8 -*-
"""
@author: Exequiel Alberto Castro Rivero
"""

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

# Terminar de rellenar este bloque con lo que vaya haciendo falta
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Importar librerías necesarias
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.utils as np_utils
from sklearn.model_selection import train_test_split

# Importar modelos y capas específicas que se van a usar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

# Importar el optimizador a usar
from tensorflow.keras.optimizers import Adam

# Importar la función de pérdida a usar
from tensorflow.keras.losses import categorical_crossentropy

# Importar el conjunto de datos
from tensorflow.keras.datasets import cifar100

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy

FAST = False


#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A esta función sólo se le llama una vez. Devuelve 4 vectores conteniendo,
# por este orden, las imágenes de entrenamiento, las clases de las imágenes
# de entrenamiento, las imágenes del conjunto de test y las clases del
# conjunto de test.

def cargarImagenes():
  # Cargamos Cifar100. Cada imagen tiene tamaño (32, 32, 3).
  # Nos vamos a quedar con las imágenes de 25 de las clases.
  
  (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  
  train_idx = np.isin(y_train, np.arange(25))
  train_idx = np.reshape(train_idx,-1)
  x_train = x_train[train_idx]
  y_train = y_train[train_idx]
  
  test_idx = np.isin(y_test, np.arange(25))
  test_idx = np.reshape(test_idx, -1)
  x_test = x_test[test_idx]
  y_test = y_test[test_idx]
  
  # Transformamos los vectores de clases en matrices. Cada componente se convierte en un vector
  # de ceros con un uno en la componente correspondiente a la clase a la que pertenece la imagen.
  # Este paso es necesario para la clasificación multiclase en keras.
  y_train = np_utils.to_categorical(y_train, 25)
  y_test = np_utils.to_categorical(y_test, 25)
  
  return x_train, y_train, x_test, y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve el accuracy de un modelo, definido como el 
# porcentaje de etiquetas bien predichas frente al total de etiquetas.
# Como parámetros es necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de keras (matrices
# donde cada etiqueta ocupa una fila, con un 1 en la posición de la clase
# a la que pertenece y 0 en las demás).

def calcularAccuracy(labels, preds):
  labels = np.argmax(labels, axis = 1)
  preds = np.argmax(preds, axis = 1)
  
  accuracy = sum(labels == preds)/len(labels)
  
  return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución de la función
# de pérdida en el conjunto de train y en el de validación, y otra
# con la evolución del accuracy en el conjunto de train y en el de
# validación. Es necesario pasarle como parámetro el historial
# del entrenamiento del modelo (lo que devuelven las funciones
# fit() y fit_generator()).

def mostrarEvolucion(hist, batch_size):

  loss = hist['loss']
  val_loss = hist['val_loss']
  plt.plot(loss)
  plt.plot(val_loss)
  plt.legend(['Training loss', 'Validation loss'])
  plt.title("Batch size = " + str(batch_size))
  plt.show()

  acc = hist['accuracy']
  val_acc = hist['val_accuracy']
  plt.plot(acc)
  plt.plot(val_acc)
  plt.legend(['Training accuracy', 'Validation accuracy'])
  plt.title("Batch size = " + str(batch_size))
  plt.show()
  

# =============================================================================
# EJERCICIO 1
# =============================================================================

#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

# Función que crea el modelo BaseNet indicado en la práctica. Recibe como
# parámetros el input_shape de las imágenes de entrada (dimensiones de la
# imagen de entrada) y el número de clases que se están tratando (num_classes).

def get_basenet(input_shape, num_classes):
    
    # El modelo BaseNet se define como un modelo secuencial, las capas añadidas
    # se irán apilando al final.
    model = Sequential()
    
    # Siguiendo la tabla de referencia se observa que la primera capa realiza
    # una convolución 2D con un kernel 5x5 sin padding (para que las dimensiones
    # del output sean 28) y con activación relu. El número de filtros a usar es
    # 6, puesto que en la tabla se especifica que el número de canales de salida
    # es 6.
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu',
                     input_shape=input_shape))
    
    # Se sigue con una MaxPooling2D para reducir a la mitad ambas dimensiones.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Se realiza otra convolución 2D, con 16 filtros.
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    
    # Otra capa MaxPooling2D
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Se realiza un "aplanado" de los datos. Lo que se consigue con esto es 
    # disponer de todos los píxeles en un array 1D para poder conectarlos
    # totalmente a las neuronas posteriores.
    model.add(Flatten())
    
    # Se mete una capa totalmente conectada con activación relu tal y como se
    # especifica en la tabla. El número de neuronas es 50.
    model.add(Dense(50, activation='relu'))

    # Se mete una capa totalmente conectada con el número de clases que se están
    # tratando y con activación 'softmax'. Esto es así porque se trata de un
    # problema de clasificación multiclase, y se usa softmax para que la salida
    # de cada neurona de esta capa sea la probabilidad de pertenecer a cada 
    # clase.
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# Justificación Adam: https://arxiv.org/pdf/1412.6980.pdf

# Compilación del modelo. Se usa entropia cruzada categórica al ser un problema
# multiclase, Adam como optimizador pos su robustez y sus buenos resultados y
# como métrica el 'accuracy' al ser una métrica típica en los problemas de
# clasificación.
def model_compile(model):
    
    # Compilación del modelo, como se trata de un problema de clasificación
    # multiclase se usa como función de pérdida la entropía cruzada categórica.
    # El optimizador usado es Adam por su conocida robustez.
    model.compile(loss=categorical_crossentropy, optimizer=Adam(),
                  metrics=['accuracy'])

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# Entrenamiento del modelo, los datos de entrenamiento son pasados por parámetros
# (x_train e y_train) además del tamaño de batch (batch_size) y el número de
# épocas (epochs). Se devuelve el historial del entrenamiento para su posterior
# visualización.
def train_model(model, x_train, y_train, batch_size, epochs):

    # Se realiza el entrenamiento con los parámetros pasados. Tal y como se
    # se indica en el enunciado, se usa un 10% de los datos para validación.
    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1)
    
    return hist

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

# Justificación del batch_size: https://arxiv.org/pdf/1606.02228.pdf

# Función que dados los datos de test (x_test e y_test) calcula la función de
# pérdida en el modelo dado (model) y el accuracy.
def get_loss_accuraccy(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    
    return score[0], score[1]


#########################################################################
################ FUNCIÓN DE EJECUCIÓN DEL EJERCICIO 1 ###################
#########################################################################

# Función que se encarga de la ejecución del ejercicio 1
def ejercicio1():
    print("------------- EJERCICIO 1 -------------\n") 
    
    # Se definen los parámetros a usar. El número de épocas se establece a 100
    # porque se considera que son suficientes para ver el comportamiento de la
    # función de pérdida y la accuracy (además de porque más épocas supone
    # más tiempo de cómputo). Para el tamaño de batch se van a probar varios
    # valores y se van a comprobar las diferencias
    epochs = 100
    batch_size = [32, 64, 128, 256]
    
    # Primero se obtienen los datos
    x_train, y_train, x_test, y_test = cargarImagenes()
    
    # Se establecen el input_shape y el número de clases (dados en el enunciado)
    input_shape = (32,32,3)
    num_classes = 25
    
    # Se obtiene el modelo
    model = get_basenet(input_shape, num_classes)
    
    # Se imprime el resumen del modelo dado
    print("Resumen del modelo diseñado\n")
    model.summary()
    input("\n--- Pulsar tecla para continuar ---\n")
    
    # Compilación del modelo
    model_compile(model)
    
    # Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los 
    # pesos aleatorios con los que empieza la red, para poder reestablecerlos 
    # después y comparar resultados entre no usar mejoras y sí usarlas.
    weights = model.get_weights()
    
    print("Experimentación con diferentes tamaños de batch\n")
    
    # Para cada tamaño de batch se realiza el entrenamiento
    for batch in batch_size:
        # Se inicializan los pesos para que no haya entrenamiento incremental
        model.set_weights(weights)
        
        print("\n--- Batch_size = " + str(batch) + "\n")
        
        # Se entrena el modelo con el batch
        hist = train_model(model, x_train, y_train, batch, epochs)
        
        print("\n------ Evolución función de pérdida y accuracy")
        
        # Se pintan los gráficos que muestran la evolución de la función de
        # pérdida y el accuracy.
        mostrarEvolucion(hist.history, batch)
        
        input("\n--- Pulsar tecla para continuar ---\n")
        
        print("------ Valores de pérdida y accuracy obtenidos en test\n\n")
        
        # Valor de pérdida y accuracy en test
        loss_test, acc_test = get_loss_accuraccy(model, x_test, y_test)
        print('Test loss:', loss_test)
        print('Accuracy test:', acc_test)
        
        input("\n--- Pulsar tecla para continuar ---\n")


# =============================================================================
# EJERCICIO 2
# =============================================================================

# =============================================================================
# MEJORA 1 -> Normalización de los datos de entrada
# =============================================================================

#########################################################################
#################### SEPARACIÓN TRAINING VALIDACIÓN #####################
#########################################################################

# Función que recibe un conjunto de datos y los separa en training/validación
# de forma que no haya clases infrarrepresentadas en ambos conjuntos. El
# porcentaje de validación es el 10%.

def split_train_val(x_train, y_train):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                            test_size=0.1, stratify=y_train)
    
    return x_train, y_train, x_val, y_val


#########################################################################
######################## NORMALIZACIÓN DE DATOS #########################
#########################################################################

# Función que recibe los conjuntos de entrenamiento y test y devuelve los iteradores
# de entrenamiento, validación y test normalizados de acuerdo a las muestras de
# entrenamiento. También recibe como parámetro el número de batches.

def normalize_images(x_train, y_train, x_test, y_test, batch_size):
    
    # Se realiza separación en training/validación
    x_train_new, y_train_new, x_val, y_val = split_train_val(x_train, y_train)
    
    # Se crea el objeto que se va encargar de normalizar los datos con los
    # parámetros correspondientes.
    datagen = ImageDataGenerator(featurewise_center = True, 
                                 featurewise_std_normalization = True)
    
    # Se ajusta a los datos de entrenamiento
    datagen.fit(x_train_new)
    
    # Se generan los iteradores
    it_train = datagen.flow(x_train_new, y_train_new, batch_size = batch_size)
    it_val = datagen.flow(x_val, y_val, batch_size = batch_size)
    it_test = datagen.flow(x_test, y_test, batch_size = batch_size)
    
    return it_train, it_val, it_test


#########################################################################
################ ENTRENAMIENTO DEL MODELO (ITERADORES) ##################
#########################################################################

# Entrenamiento del modelo, los datos de entrenamiento y validación son pasados
# por parámetros mediante iteradores (it_train, it_val) junto al número de épocas
# (epochs). Se devuelve el historial del entrenamiento para su posterior visualización.
def train_model_it(model, it_train, it_val, epochs):

    # Se realiza el entrenamiento con los parámetros pasados.
    hist = model.fit(it_train,
          epochs=epochs,
          verbose=1,
          validation_data = it_val)
    
    return hist

#########################################################################
###################### EVALUACIÓN TEST (ITERADOR) #######################
#########################################################################

# Función que dado el iterador del conjunto de test (it_test) calcula la función
# de pérdida en el modelo dado (model) y el accuracy.
def get_loss_accuraccy_it(model, it_test):
    score = model.evaluate(it_test)
    
    return score[0], score[1]

#########################################################################
################ FUNCIÓN DE EJECUCIÓN DE LA MEJORA 1 ####################
#########################################################################

# Función de ejecución de la primera mejora
def mejora1():
    print("------------- MEJORA 1\n") 
    
    # Se definen los parámetros a usar. El número de épocas se establece a 100
    # porque se considera que son suficientes para ver el comportamiento de la
    # función de pérdida y la accuracy (además de porque más épocas supone
    # más tiempo de cómputo). Para el tamaño de batch nos quedamos con 128.
    epochs = 100
    batch_size = 128
    
    # Primero se obtienen los datos
    x_train, y_train, x_test, y_test = cargarImagenes()
    
    # Se obtienen los iteradores con los datos normalizados
    it_train, it_val, it_test = normalize_images(x_train, y_train, x_test, y_test, batch_size)
    
    # Se establecen el input_shape y el número de clases (dados en el enunciado)
    input_shape = (32,32,3)
    num_classes = 25
    
    # Se obtiene el modelo
    model = get_basenet(input_shape, num_classes)
    
    # Se imprime el resumen del modelo dado
    print("Resumen del modelo diseñado\n")
    model.summary()
    input("\n--- Pulsar tecla para continuar ---\n")
    
    # Compilación del modelo
    model_compile(model)
    
    # Se entrena el modelo
    hist = train_model_it(model, it_train, it_val, epochs)
    
    print("\n------ Evolución función de pérdida y accuracy")
    
    # Se pintan los gráficos que muestran la evolución de la función de
    # pérdida y el accuracy.
    mostrarEvolucion(hist.history, batch_size)
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    print("------ Valores de pérdida y accuracy obtenidos en test\n\n")
    
    # Valor de pérdida y accuracy en test
    loss_test, acc_test = get_loss_accuraccy_it(model, it_test)
    print('Test loss:', loss_test)
    print('Accuracy test:', acc_test)
    
    input("\n--- Pulsar tecla para continuar ---\n")


# =============================================================================
# MEJORA 2 -> Introducción de Early Stopping
# =============================================================================

# Entrenamiento del modelo, los datos de entrenamiento y validación son pasados
# por parámetros mediante iteradores (it_train, it_val) junto al número de épocas
# (epochs). Se devuelve el historial del entrenamiento para su posterior visualización.
# El entrenamiento se realiza con Early Stopping.
def train_model_early(model, it_train, it_val, epochs):
    
    # Se define el callback para implementar el Early Stopping durante el
    # entrenamiento. El early stopping se realiza sobre la función de pérdida
    # en el conjunto de validación (tal y como se ha visto en AA, llegado el
    # momento en el que el error en validación sube es mejor parar). El número
    # de épocas en las que no tiene que haber mejora se establece a 5 por 
    # experimentación. Los mejores pesos encontrados se restauran.
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                                restore_best_weights = True)

    # Se realiza el entrenamiento con los parámetros pasados. Tal y como se
    # se indica en el enunciado, se usa un 10% de los datos para validación.
    hist = model.fit(it_train,
          epochs=epochs,
          verbose=1,
          validation_data = it_val,
          callbacks = [callback])
    
    return hist


#########################################################################
################ FUNCIÓN DE EJECUCIÓN DE LA MEJORA 2 ####################
#########################################################################

# Función de ejecución de la mejora 2.
def mejora2():
    print("------------- MEJORA 2\n") 
    
    # Se definen los parámetros a usar. El número de épocas se establece a 100
    # porque se considera que son suficientes para ver el comportamiento de la
    # función de pérdida y la accuracy (además de porque más épocas supone
    # más tiempo de cómputo). Para el tamaño de batch nos quedamos con 128.
    epochs = 100
    batch_size = 128
    
    # Primero se obtienen los datos
    x_train, y_train, x_test, y_test = cargarImagenes()
    
    # Se obtienen los iteradores con los datos normalizados
    it_train, it_val, it_test = normalize_images(x_train, y_train, x_test, y_test, batch_size)
    
    # Se establecen el input_shape y el número de clases (dados en el enunciado)
    input_shape = (32,32,3)
    num_classes = 25
    
    # Se obtiene el modelo
    model = get_basenet(input_shape, num_classes)
    
    # Se imprime el resumen del modelo dado
    print("Resumen del modelo diseñado\n")
    model.summary()
    input("\n--- Pulsar tecla para continuar ---\n")
    
    # Compilación del modelo
    model_compile(model)
    
    # Se entrena el modelo
    hist = train_model_early(model, it_train, it_val, epochs)
    
    print("\n------ Evolución función de pérdida y accuracy")
    
    # Se pintan los gráficos que muestran la evolución de la función de
    # pérdida y el accuracy.
    mostrarEvolucion(hist.history, batch_size)
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    print("------ Valores de pérdida y accuracy obtenidos en test\n\n")
    
    # Valor de pérdida y accuracy en test
    loss_test, acc_test = get_loss_accuraccy_it(model, it_test)
    print('Test loss:', loss_test)
    print('Accuracy test:', acc_test)
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
# =============================================================================
# MEJORA 3 -> Data augmentation
# =============================================================================

# Función que recibe los conjuntos de entrenamiento y test y devuelve un iterador
# de entrenamiento con datos aumentados y normalizados, además de los iteradores
# validación y test normalizados de acuerdo a las muestras de entrenamiento.
# También recibe como parámetro el número de batches. El data augmentation no
# debe de ser muy agresivo debido a que podría haber un empeoramiento del
# rendimiento.

def data_augmentation_normalized(x_train, y_train, x_test, y_test, batch_size):
    
    # Se realiza separación en training/validación
    x_train_new, y_train_new, x_val, y_val = split_train_val(x_train, y_train)
    
    # Se crea el objeto que se va a encargar de normalizar y aumentar los datos
    # de entrenamiento. No debe de ser muy agresivo para no distorsinar demasiado
    # los datos. En concreto, se tomarán rotaciones y zoom con unos valores no
    # demasiado grandes para no obtener un empeoramiento drástico.
    datagenTrain = ImageDataGenerator(featurewise_center = True, 
                                 featurewise_std_normalization = True,
                                 rotation_range = 15, zoom_range = 0.2,
                                 horizontal_flip = True)
    
    # Se crea el objeto que se va encargar de normalizar los datos con los
    # parámetros correspondientes (para test y validación)
    datagenNotTrain = ImageDataGenerator(featurewise_center = True, 
                                 featurewise_std_normalization = True)
    
    # Se ajustan a los datos de entrenamiento
    datagenTrain.fit(x_train_new)
    datagenNotTrain.fit(x_train_new)
    
    # Se generan los iteradores
    it_train = datagenTrain.flow(x_train_new, y_train_new, batch_size = batch_size)
    it_val = datagenNotTrain.flow(x_val, y_val, batch_size = batch_size)
    it_test = datagenNotTrain.flow(x_test, y_test, batch_size = batch_size)
    
    return it_train, it_val, it_test


#########################################################################
################ FUNCIÓN DE EJECUCIÓN DE LA MEJORA 3 ####################
#########################################################################

# Función de ejecución de la mejora 3.
def mejora3():
    print("------------- MEJORA 3\n") 
    
    # Se definen los parámetros a usar. El número de épocas se establece a 100
    # porque se considera que son suficientes para ver el comportamiento de la
    # función de pérdida y la accuracy (además de porque más épocas supone
    # más tiempo de cómputo). Para el tamaño de batch nos quedamos con 128.
    epochs = 100
    batch_size = 128
    
    # Primero se obtienen los datos
    x_train, y_train, x_test, y_test = cargarImagenes()
    
    # Se obtienen los iteradores con los datos normalizados
    it_train, it_val, it_test = data_augmentation_normalized(x_train, y_train,
                                                             x_test, y_test,
                                                             batch_size)
    
    # Se establecen el input_shape y el número de clases (dados en el enunciado)
    input_shape = (32,32,3)
    num_classes = 25
    
    # Se obtiene el modelo
    model = get_basenet(input_shape, num_classes)
    
    # Se imprime el resumen del modelo dado
    print("Resumen del modelo diseñado\n")
    model.summary()
    input("\n--- Pulsar tecla para continuar ---\n")
    
    # Compilación del modelo
    model_compile(model)
    
    # Se entrena el modelo
    hist = train_model_early(model, it_train, it_val, epochs)
    
    print("\n------ Evolución función de pérdida y accuracy")
    
    # Se pintan los gráficos que muestran la evolución de la función de
    # pérdida y el accuracy.
    mostrarEvolucion(hist.history, batch_size)
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    print("------ Valores de pérdida y accuracy obtenidos en test\n\n")
    
    # Valor de pérdida y accuracy en test
    loss_test, acc_test = get_loss_accuraccy_it(model, it_test)
    print('Test loss:', loss_test)
    print('Accuracy test:', acc_test)
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    tf.keras.backend.clear_session()
    
# =============================================================================
# MEJORA 4 -> Aumento de la profundidad del modelo
# =============================================================================

# Función que crea el modelo BaseNet ampliado. Recibe como
# parámetros el input_shape de las imágenes de entrada (dimensiones de la
# imagen de entrada) y el número de clases que se están tratando (num_classes).

def get_deep_basenet(input_shape, num_classes):
    
    # El modelo BaseNet se define como un modelo secuencial, las capas añadidas
    # se irán apilando al final.
    model = Sequential()
    
    # Se realiza una convolución transpuesta para no perder demasiada dimensionalidad.
    # Además, se le da más importancia a las características centrales.
    model.add(Conv2DTranspose(6, kernel_size=(3, 3), activation='relu',
                              input_shape=input_shape))
    
    # Se realiza una convolución transpuesta para no perder demasiada dimensionalidad.
    # Además, se le da más importancia a las características centrales.
    model.add(Conv2DTranspose(6, kernel_size=(3, 3), activation='relu'))
    
    # Siguiendo la tabla de referencia se observa que la primera capa realiza
    # una convolución 2D con un kernel 5x5 sin padding (para que las dimensiones
    # del output sean 28) y con activación relu. El número de filtros a usar es
    # 6, puesto que en la tabla se especifica que el número de canales de salida
    # es 6.
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu'))
    
    # Se realiza una convolución con 6 filtros de tamaño 3x3.
    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu'))
    
    # Se realiza otra convolución 2D de 3x3 con 6 filtros
    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu'))
    
    # Se sigue con una MaxPooling2D para reducir a la mitad ambas dimensiones.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Se realiza una convolución transpuesta para no perder demasiada dimensionalidad.
    # Además, se le da más importancia a las características centrales.
    model.add(Conv2DTranspose(16, kernel_size=(3, 3), activation='relu'))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 4 filtros. 
    model.add(Conv2D(4, kernel_size=(1, 1), activation=LeakyReLU()))

    # Se realiza otra convolución 2D, con 16 filtros.
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 4 filtros. 
    model.add(Conv2D(4, kernel_size=(1, 1), activation='relu'))
    
    # Se realiza otra convolución 2D de 3x3 con 6 filtros
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    
    # Otra capa MaxPooling2D
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Se realiza un "aplanado" de los datos. Lo que se consigue con esto es 
    # disponer de todos los píxeles en un array 1D para poder conectarlos
    # totalmente a las neuronas posteriores.
    model.add(Flatten())
    
    # Se mete una capa totalmente conectada con activación relu tal y como se
    # especifica en la tabla. El número de neuronas es 50.
    model.add(Dense(50, activation='relu'))

    # Se mete una capa totalmente conectada con el número de clases que se están
    # tratando y con activación 'softmax'. Esto es así porque se trata de un
    # problema de clasificación multiclase, y se usa softmax para que la salida
    # de cada neurona de esta capa sea la probabilidad de pertenecer a cada 
    # clase.
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
    
    # El modelo BaseNet se define como un modelo secuencial, las capas añadidas
    # se irán apilando al final.
    model = Sequential()
    
    # Se realiza una convolución transpuesta para no perder demasiada dimensionalidad.
    # Además, se le da más importancia a las características centrales.
    model.add(Conv2DTranspose(64, kernel_size=(3, 3), activation=LeakyReLU(),
                              input_shape=input_shape))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 8 filtros. 
    model.add(Conv2D(16, kernel_size=(1, 1), activation=LeakyReLU()))
    
    # Se realiza una convolución transpuesta para no perder demasiada dimensionalidad.
    # Además, se le da más importancia a las características centrales.
    model.add(Conv2DTranspose(64, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 8 filtros. 
    model.add(Conv2D(16, kernel_size=(1, 1), activation=LeakyReLU()))
    
    # Se realiza una convolución inicial con 16 filtros de tamaño 3x3. La
    # activación es Leaky Relu.
    model.add(Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 8 filtros. 
    model.add(Conv2D(16, kernel_size=(1, 1), activation=LeakyReLU()))
    
    # Se realiza otra convolución 2D de 3x3 con 16 filtros
    model.add(Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 8 filtros. 
    model.add(Conv2D(16, kernel_size=(1, 1), activation=LeakyReLU()))
    
    # Se realiza otra convolución 2D de 3x3 con 16 filtros
    model.add(Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 8 filtros. 
    model.add(Conv2D(16, kernel_size=(1, 1), activation=LeakyReLU()))
    
    # Se realiza otra convolución 2D de 3x3 con 16 filtros
    model.add(Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Se sigue con una MaxPooling2D para reducir a la mitad ambas dimensiones.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Se realiza una convolución transpuesta para no perder demasiada dimensionalidad.
    # Además, se le da más importancia a las características centrales.
    model.add(Conv2DTranspose(128, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 8 filtros. 
    model.add(Conv2D(32, kernel_size=(1, 1), activation=LeakyReLU()))

    # Se realiza otra convolución 2D 3x3, con 32 filtros.
    model.add(Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 8 filtros. 
    model.add(Conv2D(32, kernel_size=(1, 1), activation=LeakyReLU()))
    
    # Se realiza otra convolución 2D 3x3, con 32 filtros.
    model.add(Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Se realiza una convolución 1x1 para que las convoluciones siguientes no
    # sean tan pesadas. La convolución se realiza con 8 filtros. 
    model.add(Conv2D(32, kernel_size=(1, 1), activation=LeakyReLU()))
    
    # Se realiza otra convolución 2D 3x3, con 32 filtros.
    model.add(Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU()))
    
    # Otra capa MaxPooling2D
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Se realiza un "aplanado" de los datos. Lo que se consigue con esto es 
    # disponer de todos los píxeles en un array 1D para poder conectarlos
    # totalmente a las neuronas posteriores.
    model.add(Flatten())
    
    # Se mete una capa totalmente conectada con activación LeakyRelu y 200 neuronas.
    model.add(Dense(200, activation=LeakyReLU()))

    # Se mete una capa totalmente conectada con el número de clases que se están
    # tratando y con activación 'softmax'. Esto es así porque se trata de un
    # problema de clasificación multiclase, y se usa softmax para que la salida
    # de cada neurona de esta capa sea la probabilidad de pertenecer a cada 
    # clase.
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


#########################################################################
################ FUNCIÓN DE EJECUCIÓN DE LA MEJORA 4 ####################
#########################################################################

# Función de ejecución de la mejora 3.
def mejora4():
    print("------------- MEJORA 4: Profundización de la red\n") 
    
    # Se definen los parámetros a usar. El número de épocas se establece a 100
    # porque se considera que son suficientes para ver el comportamiento de la
    # función de pérdida y la accuracy (además de porque más épocas supone
    # más tiempo de cómputo). Para el tamaño de batch nos quedamos con 128.
    epochs = 100
    batch_size = 128
    
    # Primero se obtienen los datos
    x_train, y_train, x_test, y_test = cargarImagenes()
    
    # Se obtienen los iteradores con los datos normalizados
    it_train, it_val, it_test = data_augmentation_normalized(x_train, y_train,
                                                             x_test, y_test,
                                                             batch_size)
    
    # Se establecen el input_shape y el número de clases (dados en el enunciado)
    input_shape = (32,32,3)
    num_classes = 25
    
    # Se obtiene el modelo
    model = get_deep_basenet(input_shape, num_classes)
    
    # Se imprime el resumen del modelo dado
    print("Resumen del modelo diseñado\n")
    model.summary()
    input("\n--- Pulsar tecla para continuar ---\n")
    
    # Compilación del modelo
    model_compile(model)
    
    # Se entrena el modelo
    hist = train_model_early(model, it_train, it_val, epochs)
    
    print("\n------ Evolución función de pérdida y accuracy")
    
    # Se pintan los gráficos que muestran la evolución de la función de
    # pérdida y el accuracy.
    mostrarEvolucion(hist.history, batch_size)
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    print("------ Valores de pérdida y accuracy obtenidos en test\n\n")
    
    # Valor de pérdida y accuracy en test
    loss_test, acc_test = get_loss_accuraccy_it(model, it_test)
    print('Test loss:', loss_test)
    print('Accuracy test:', acc_test)
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    tf.keras.backend.clear_session()