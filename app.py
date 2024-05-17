import streamlit as st  # importing streamlit and tensorflow
import tensorflow as tf
import cv2
import keras
from tensorflow.keras.models import Sequential, load_model  # Import load_model directly
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
from PIL import Image, ImageOps
from model import load_custom_model  # Import the renamed function

st.set_option('deprecation.showfileUploaderEncoding', False)  # Prevent warning on loading streamlit app

@st.cache(allow_output_mutation=True)  # Cache the model to avoid reloading
def load_model_from_file():  # Loading our model
    model = load_model('/content/model.h5')  # Use TensorFlow's load_model
    return model

model = load_model_from_file()

# Defining the header or title of the page that the user will be seeing
st.markdown("<h1 style='text-align: center; color: Black;'>Brain Tumor Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: Black;'>All you have to do is Upload the MRI scan and the model will do the rest!</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: Black;'>Submission for HackOff V3.0</h4>", unsafe_allow_html=True)
st.sidebar.header("What is this Project about?")
st.sidebar.text("It is a Deep learning solution to detection of Brain Tumor using MRI Scans.")
st.sidebar.header("What does it do?")
st.sidebar.text("The user can upload their MRI scan and the model will try to predict whether or not the user has Brain Tumor or not.")
st.sidebar.header("What tools were used to make this?")
st.sidebar.text("The Model was made using a dataset from Kaggle along with using Kaggle notebooks to train the model. We made use of Tensorflow, Keras as well as some other Python Libraries to make this complete project. To deploy it on web, we used ngrok and Streamlit!")

file = st.file_uploader("Please upload your MRI Scan", type=["jpg", "png"])  # Accepting the image input from the user

def import_and_predict(image_data, model):  # Prediction method accepting data and model
    size = (150, 150)
    image1 = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = ImageOps.grayscale(image1)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    img_reshape = img.reshape(1, 150, 150, 1)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.markdown("<h5 style='text-align: center; color: Black;'>Please Upload a File</h5>", unsafe_allow_html=True)
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Stage_3_Tumor', 'Stage_2_Tumor', 'no_tumor', 'Stage_1_Tumor']
    string = "The patient most likely has:" + class_names[np.argmax(predictions)]
    st.success(string)
