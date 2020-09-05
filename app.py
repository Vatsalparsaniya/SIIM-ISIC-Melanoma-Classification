import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn

import streamlit as st
from PIL import Image, ImageOps

MODEL = 'model/BESTAUC_WEIGHTS_B4_512_FOLD1.hdf5'

IMG_N = int(MODEL.split('_')[3])
EFF_NET = MODEL.split('_')[2]

def getModel():
  modelName = f'EfficientNet{EFF_NET}'
    
  constructor = getattr(efn, modelName)
        
  base_model = constructor(include_top=False,
                           weights='noisy-student',
                           input_shape=(IMG_N, IMG_N, 3))
    
  x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

  x = tf.keras.layers.Dense(1,activation='sigmoid',dtype='float32')(x)

  model = tf.keras.Model(inputs=base_model.input,outputs=x,name=modelName)

  model.load_weights(MODEL)

  return model


def import_and_predict(image_data, model, size=(IMG_N,IMG_N)):
       
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  image = np.asarray(image).astype(np.float32)
  
  image = image/255.

  image = image[np.newaxis,...]

  prediction = model.predict(image)

  return prediction[0][0]

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("🔬 SIIM-ISIC Melanoma Classification")
st.header("🔍 Identify melanoma in Skin-lesion images")

st.write("This is a simple image classification web app to classify between malignant and benign images")
file = st.file_uploader("Please upload a skin-lesion image for classification: ", type=["jpg", "png"])

model = getModel()

if file is not None:
  image = Image.open(file)
  st.image(image, use_column_width=True)
  prediction = import_and_predict(image, model)

  st.text("Probability [0: benign, 1: malignant]")
  st.write("Prediction: ",prediction)