# Melanoma Detection Convolutional Neural Network
This project focuses on the development of a convolutional neural network (CNN) using TensorFlow's Keras to detect melanoma, a form of skin cancer. The CNN is trained to classify skin lesions as either benign or malignant based on images.

## Project Overview:
This project enables the early detection of melanoma, which can be a life-saving application of deep learning in the field of healthcare. 
The CNN model can classify skin lesions with a high degree of accuracy.

## What is Melanoma?
Melanoma is a deadly form of skin cancer, and early detection is crucial for successful treatment. 
In this project, we leverage deep learning to create a model that can classify skin lesions as benign or malignant. 

## Code Features:

### Data Preprocessing
We begin by collecting the file paths of images from two folders, one containing malignant skin lesions and the other containing benign skin lesions. The file paths are stored in lists for further processing.

### Making Labels for the Data
We assign labels to the images, where 1 represents malignant and 0 represents benign. These labels are used for supervised training of the CNN.

### Processing and Converting Images to Numpy Arrays
We resize and convert the images to numpy arrays, making them suitable for training.

### Train-Test Split
We split the data into training and testing sets using Scikit Learn's splitting function.

### Scaling for Easier Training
We scale the image data to values between 0 and 1 to facilitate training.

### Building the CNN Model
The CNN model is designed to capture features from images for classification. It consists of convolutional layers, pooling layers, and fully connected layers.

### Compiling the Model
We compile the model by specifying the optimizer, loss function, and metrics.

### Training the Model
The model is trained on the training data.

### Model Evaluation
We evaluate the model's performance on the test data.

### Predicting Using the Trained Model
You can use the trained model to make predictions on new images. Provide the full path to the image, and the model will classify it as malignant or benign.
