
import pandas as pd, numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("""
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
            """,unsafe_allow_html=True)

st.markdown("""
              <div class="container">
                <div class="alert alert-info">
                  <strong>Info!</strong> Because of space limitation[500 MB] we have deployed on Heroku webapp only Model and Tensorflow exceeds limit of 500MB.
                </div>
              </div>
            """,unsafe_allow_html=True)

st.markdown("""
              <div class="container">
                <div class="alert alert-info">
                  <strong>Info!</strong> Try these combinations <br> Efficientnet B1-384<br> Efficientnet B3-768<br> Efficientnet B4-768<br> Efficientnet B5-768<br> Efficientnet B6-512
                </div>
              </div>
            """,unsafe_allow_html=True)


IMG_SIZES = 768
EFF_NETS = 5

# EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
#         efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

# def build_model(dim=128, ef=0):
#     inp = tf.keras.layers.Input(shape=(dim,dim,3))
#     base = EFNS[ef](input_shape=(dim,dim,3),weights='imagenet',include_top=False)
#     x = base(inp)
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
#     model = tf.keras.Model(inputs=inp,outputs=x)
#     return model

# def import_and_predict(image_data, model, size=(IMG_SIZES,IMG_SIZES)):

#     image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#     image = np.asarray(image).astype(np.float32)/255.
#     image = np.expand_dims(image, axis=0)
#     prediction = model.predict(image)
    
#     return prediction

st.title('SIIM-ISIC Melanoma Classification Prediction')
st.markdown("Github Repository [Click Here](https://github.com/Vatsalparsaniya/SIIM-ISIC-Melanoma-Classification)")
st.markdown("---")
st.markdown("<img src='https://raw.githubusercontent.com/Vatsalparsaniya/SIIM-ISIC-Melanoma-Classification/master/reports/figures/banner.png' alt='Banner' width='100%'/>",unsafe_allow_html=True)
st.markdown("---")

Model_list = ['Efficientnet B0','Efficientnet B1','Efficientnet B2',
              'Efficientnet B3','Efficientnet B4','Efficientnet B5',
              'Efficientnet B6','Efficientnet B7']
selected_model = st.sidebar.radio("Select Efficientnet Model :", Model_list)
EFF_NETS = Model_list.index(selected_model)


Image_size_list = [128, 192, 256, 384, 512, 768]
IMG_SIZES = st.sidebar.radio("Select Image Size :", Image_size_list) 


st.markdown(f"#### Selected Efficientnet Model in sidebar : <span style='color:blue'>{selected_model}</span>",unsafe_allow_html=True)
st.markdown(f"#### Selected Image-Size in sidebar : <span style='color:blue'>{IMG_SIZES}</span>",unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"## <span style='color:black'>Trained Model {selected_model} - {IMG_SIZES} discripction : </span>",unsafe_allow_html=True)



if os.path.isfile(f'models/B{EFF_NETS}-{IMG_SIZES}/train_log.csv'):
    st.markdown("""
              <div class="container">
                <div class="alert alert-success">
                  <strong>Success!</strong> Model Found.
                </div>
              </div>
            """,unsafe_allow_html=True)
    st.write(f"#### <span style='color:black'>Training Log</span>",unsafe_allow_html=True)
    history_data = pd.read_csv(f'models/B{EFF_NETS}-{IMG_SIZES}/train_log.csv')
    EPOCHS = len(history_data['epoch'])
    history = pd.DataFrame({'history':history_data.to_dict('list')})
    fig = plt.figure(figsize=(15,5))
    plt.plot(np.arange(EPOCHS),history.history['auc'],'-o',label='Train AUC',color='#ff7f0e')
    plt.plot(np.arange(EPOCHS),history.history['val_auc'],'-o',label='Val AUC',color='#1f77b4')
    x = np.argmax( history.history['val_auc'] ); y = np.max( history.history['val_auc'] )
    st.markdown(f"#### Max AUC : <span style='color:green'>{y}</span>",unsafe_allow_html=True)
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)
    plt.ylabel('AUC',size=14); plt.xlabel('Epoch',size=14)
    plt.legend(loc=2)
    plt2 = plt.gca().twinx()
    plt2.plot(np.arange(EPOCHS),history.history['loss'],'-o',label='Train Loss',color='#2ca02c')
    plt2.plot(np.arange(EPOCHS),history.history['val_loss'],'-o',label='Val Loss',color='#d62728')
    x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )
    st.markdown(f"#### Min Loss : <span style='color:green'>{y}</span>",unsafe_allow_html=True)
    ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
    plt.ylabel('Loss',size=14)
    plt.title(f" Model : {selected_model} , Image-Size : {IMG_SIZES} ",size=18)
    plt.legend(loc=3)
    st.write(fig)
    st.markdown("---")

    # model = build_model(dim=IMG_SIZES, ef=EFF_NETS)
    st.header("🔍 Identify melanoma in Skin-lesion images")
    file = st.file_uploader("Please upload a skin-lesion image for classification: ", type=["jpg", "png"])

    if file is not None:
      image = Image.open(file)
      st.image(image, use_column_width=True)

      # prediction = import_and_predict(image, model,size=(IMG_SIZES,IMG_SIZES))

      # st.text("Probability [0: benign, 1: malignant]")
      # st.write("Prediction: ",prediction[0][0])
      st.markdown("""
              <div class="container">
                <div class="alert alert-danger">
                  <strong>Sorry!</strong>Model not available on Heroku Check our Repository<br>
                  Select diffrent Model and Image-Size from sidebar.
                </div>
              </div>
            """,unsafe_allow_html=True)

else:
  st.markdown("""
              <div class="container">
                <div class="alert alert-danger">
                  <strong>Sorry!</strong> specified combination of model is not deployed here.<br>
                  Select diffrent Model and Image-Size from sidebar.
                </div>
              </div>
            """,unsafe_allow_html=True)
  
