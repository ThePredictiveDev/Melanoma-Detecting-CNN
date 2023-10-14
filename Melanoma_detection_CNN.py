#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
import os


# # Data Preprocessing

# ## Storing File Paths from Folder into a list

# In[3]:


malignant_files = os.listdir("C:/Users/Devansh/Downloads/Cancer detection deep learning project/melanoma_cancer_dataset/malignant")
benign_files = os.listdir("C:/Users/Devansh/Downloads/Cancer detection deep learning project/melanoma_cancer_dataset/benign")
print(malignant_files[0:5]) #first five file names in malignant 
print(malignant_files[-5:]) #last five file names in malignant 
print(benign_files[0:5]) #first five file names in benign 
print(benign_files[-5:]) #last five file names in benign 


# In[4]:


malignant_cases_number = len(malignant_files)
benign_cases_number = len(benign_files)


# ## Making Labels for the Data

# In[5]:


malignant_labels = [1]*malignant_cases_number
benign_labels = [0]*benign_cases_number
labels = malignant_labels+benign_labels


# In[6]:


malignant_path = "C:/Users/Devansh/Downloads/Cancer detection deep learning project/melanoma_cancer_dataset/malignant/"
benign_path = "C:/Users/Devansh/Downloads/Cancer detection deep learning project/melanoma_cancer_dataset/benign/"


# ## Processing and Converting Image to Numpy Array for Training

# In[16]:


data = []
for img_file in malignant_files:
    image = Image.open(malignant_path + img_file) #opens all the images
    image = image.convert('RGB') #converts to rgb
    image = image.resize((128,128)) # resizes to 128x128
    image = np.array(image) #converts to numpy array
    data.append(image) 
    
for img_file in benign_files:
    image = Image.open(benign_path + img_file) #opens all the images
    image = image.convert('RGB') #converts to rgb
    image = image.resize((128,128)) # resizes to 128x128
    image = np.array(image) #converts to numpy array
    data.append(image)


# In[17]:


X = np.array(data) #converts data to numpy array
Y = np.array(labels) #converts labels to numpy array


# # Train Test Split using Scikit Learn's Splitting Function

# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1 ,random_state=2)


# # Scaling for Easier Training

# In[19]:


X_train = X_train/255
X_test = X_test/255


# # Building, Compiling and Training the Model Using Tensorflow (Keras)!

# ## The model will look something like this:
# 
# 
# 
# 
# 

# ![Neural%20Network%20Representation.png](attachment:Neural%20Network%20Representation.png)

# In[21]:


num_of_classes = 2

model = keras.Sequential([
    
                    keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)),
                    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
                    keras.layers.Conv2D(64, kernel_size=(2,2), activation='relu'),
                    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
                    keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu'),
                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                        
                    keras.layers.Flatten(),
                    keras.layers.Dense(128, activation = 'relu'),
                    keras.layers.Dropout(0.5),
                
                
                    keras.layers.Dense(64, activation = 'relu'),    
                    keras.layers.Dropout(0.5),
                    
                    keras.layers.Dense(num_of_classes, activation = 'sigmoid'),
                        
                        
                        ])


# ## Compiling the Model

# In[22]:


model.compile(optimizer='adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['acc'])


# ## Training the Model

# In[24]:


history = model.fit(X_train, Y_train, validation_split=0.1, epochs = 50)


# ## Checking the accuracy of the model

# In[26]:


loss, accuracy = model.evaluate(X_test, Y_test)
print("Accuracy = ", accuracy*100,"%")
print("Loss is: ", loss)


# # Plotting Various Graphs Pertaining to Loss and Accuracy Throughout Training Time

# In[34]:


h = history
plt.plot(h.history["loss"], label = 'Train Loss')
plt.plot(h.history["acc"], label = 'Train Accuracy')
plt.title("Loss vs Accuracy for Training data")
plt.show()

plt.plot(h.history["val_loss"], label = 'Train Loss')
plt.plot(h.history["val_acc"], label = 'Train Accuracy')
plt.title("Loss vs Accuracy for Validation data")
plt.show()

plt.plot(h.history["acc"], label = 'Train Accuracy')
plt.plot(h.history["val_acc"], label = 'Train Accuracy')
plt.title("Training vs Validation Accuracy")
plt.show()

plt.plot(h.history["loss"], label = 'Train Loss')
plt.plot(h.history["val_loss"], label = 'Train Loss')
plt.title("Training vs Validation Loss")
plt.show()



# # Predicting Engine

# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
import time 

input_image_path = input("Input path for image to be predicted(full path C:): ")

input_Image = Image.open(input_image_path)
input_image = input_Image.resize((128,128))
input_image = np.array(input_image)
input_image = input_image/255
input_image = np.reshape(input_image, [1,128,128,3])
input_prediction = model.predict(input_image)

input_pred_label = np.argmax(input_prediction)
if input_pred_label == 1:
    print("---------------------------------------------------")
    print("THIS TUMOR IS MALIGNANT WITH A 90% CERTAINTY")
    print("---------------------------------------------------")

elif input_pred_label == 0:
    print("---------------------------------------------------")
    print("THIS TUMOR IS BENIGN WITH A 90% CERTAINTY")
    print("---------------------------------------------------")

img = mpimg.imread(input_image_path)
imgplot=plt.imshow(img)
plt.title("THIS IS THE INPUT IMAGE")
plt.show()

