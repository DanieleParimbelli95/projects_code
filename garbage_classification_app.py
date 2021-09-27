import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from urllib.request import urlopen
from urllib.error import HTTPError
from PIL import Image
import sys

sys.tracebacklimit = 0

st.title('Garbage Classification App')

author = "*Created by* [*Daniele Parimbelli*](https://danieleparimbelli95.github.io/)"
st.markdown(author)

st.sidebar.header('How Was the App Made?')
about = 'Using a pretrained model, which was then fine-tuned on images downloaded from [here](https://www.kaggle.com/asdasdasasdas/garbage-classification) (data augmentation was used).'
st.sidebar.markdown(about)

st.sidebar.header('Use Case Example')
st.sidebar.write('A model like this could be useful for a smart recycling system based on the recognition of waste materials.')

st.sidebar.header('Possible Improvements')
st.sidebar.write('The model was trained on my PC, so it definitely has some limitations.')
st.sidebar.write('An obvious way to improve accuracy would be to use a machine with more computational power, so that:')

improvements1 = '* the number of images used during training can be increased;'
st.sidebar.markdown(improvements1)
improvements2 = '* training can last longer.'
st.sidebar.markdown(improvements2)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 335px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 335px;
        margin-left: -335px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write('The app classifies pictures of objects (bottles, cans, boxes...) into one of the following categories: Plastic, Glass, Paper, Cardboard, Metal.')

@st.cache(allow_output_mutation = True, show_spinner = False, suppress_st_warning = True)
def load_model():
    model = keras.models.load_model("EfficientNetB6.h5")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = [keras.metrics.AUC(curve = "PR")])
    return model

img_size = 384

with st.expander("Recognize material from an image file"):
    
    uploaded_file = st.file_uploader("Upload an image", type = ['jpg', 'jpeg','png', 'bmp', 'gif'])

    if uploaded_file is not None:
        
        @st.cache(allow_output_mutation = True, show_spinner = False, suppress_st_warning = True)
        def classify_from_file():
            
            img = Image.open(uploaded_file).convert('RGB').resize((img_size, img_size))
            img_array = image.img_to_array(img)
            img_to_predict = np.expand_dims(img_array, axis = 0)   
            
            model = load_model()
            
            pred_array = model.predict(img_to_predict)
            pred_label = str(np.argmax(pred_array))
            pred_prob = "{:.2%}".format(np.max(pred_array))
            pred_class_array = np.select([pred_label == '0', pred_label == '1', pred_label == '2', pred_label == '3', pred_label == '4'], ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic'])
            pred_class = np.array2string(pred_class_array).replace("'", "")
            
            img_to_print = Image.open(uploaded_file).convert('RGB').resize((img_size / 2, img_size / 2))
            
            return pred_class, pred_prob, img_to_print
                
        st.image(classify_from_file()[2])
        st.write('**Prediction: **', classify_from_file()[0])
        st.write('**Probability: **', classify_from_file()[1])
        

with st.expander("Recognize material from an image url"):
    
    url = st.text_input("Enter a valid url")
    
    if url != '':
        
        try:
            img = Image.open(urlopen(url)).convert('RGB').resize((img_size, img_size))
        except HTTPError:
            st.error('This url is forbidden')
            st.stop()
        except:
            st.error('Please enter a valid url')
            st.stop()
        
        @st.cache(allow_output_mutation = True, show_spinner = False, suppress_st_warning = True)
        def classify_from_url():
            
            img_array = image.img_to_array(img)  
            img_to_predict = np.expand_dims(img_array, axis = 0)   
            
            model = load_model()
        
            pred_array = model.predict(img_to_predict)
            pred_label = str(np.argmax(pred_array))
            pred_prob = "{:.2%}".format(np.max(pred_array))
            pred_class_array = np.select([pred_label == '0', pred_label == '1', pred_label == '2', pred_label == '3', pred_label == '4'], ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic'])
            pred_class = np.array2string(pred_class_array).replace("'", "")
            
            img_to_print = Image.open(urlopen(url)).convert('RGB').resize((img_size / 2, img_size / 2))
            
            return pred_class, pred_prob, img_to_print
        
        st.image(classify_from_url()[2])
        st.write('**Prediction: **', classify_from_url()[0])
        st.write('**Probability: **', classify_from_url()[1])
  