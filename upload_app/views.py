from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from django.db import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


# Create your views here.
def hotel_image_view(request):
    if request.method == 'POST':
        form = HotelForm(request.POST, request.FILES)
  
        if form.is_valid():

            

            # uploaded_file = request.FILES['hotel_Main_Img']
            # str_text = ''
            # for line in uploaded_file:
            #     str_text = line.decode('utf-8')

            # fs = FileSystemStorage()
            last_record = Hotel.objects.latest('hotel_Main_Img')
            path = '/Users/dastanarysbay/Desktop/file_project/media/images/3dwall66_rejAMGP.jpg'
        img = image.load_img(path)
        train = ImageDataGenerator(rescale = 1/255)
        validation = ImageDataGenerator(rescale = 1/255)
        train_dataset = train.flow_from_directory('/Users/dastanarysbay/Desktop/file_project/media/',
                                         target_size = (200,200),
                                         batch_size  = 3, 
                                         class_mode= 'binary')
        validation_dataset = validation.flow_from_directory('/Users/dastanarysbay/Desktop/file_project/validation/',
                                         target_size = (200,200),
                                         batch_size  = 3, 
                                         class_mode= 'binary')
        model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32, (3,3), activation= 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation = 'relu'), 
                                    ##
                                    tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(loss= 'binary_crossentropy',
             optimizer = RMSprop(lr=0.001),
             metrics= ['accuracy'])  
        history = model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 10,
                     validation_data= validation_dataset)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        test_loss, test_acc = model.evaluate(validation_dataset)
        print(test_acc)
        # test_acc = form.cleaned_data['result']
        # reg = Hotel(result=test_acc)
        form.save()
        return redirect('success')
    else:
        form = HotelForm()
    return render(request, 'index.html', {'form' : form})
  
  
def success(request):
    return render(request, 'button.html')

