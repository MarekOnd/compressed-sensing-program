import streamlit as st
import pandas as pd
import argparse as ap
import datetime as dt
import numpy as np
from PIL import Image

def to_img(arr):
    return Image.fromarray(np.uint8(arr))

def main(args):
    st.set_page_config(page_title = "Streamlit app", layout='wide')
    st.title('Interactive app')
    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img)
        array = np.asarray(img)
        #st.data_editor(array[:,:,0])
        #st.data_editor(array[:,:,1])
        #st.data_editor(array[:,:,2])

        black_and_white = np.round(0.3*array[:,:,0] + 0.6*array[:,:,1] + 0.11*array[:,:,2])
        st.data_editor(black_and_white)
        st.image(Image.fromarray(np.uint8(black_and_white)))

        fouriered = np.fft.fft2(black_and_white)
        angles = np.angle(fouriered)
        st.data_editor(np.uint8((angles)/360*256))
        st.image(Image.fromarray(np.uint8((angles)/360*256)),width=1000)


        a = st.slider("Some value",0,100,10)
        st.text(f"{a}")
        m = a
        fouriered[fouriered.shape[0]-m:fouriered.shape[0],fouriered.shape[1]-m:fouriered.shape[1]] = np.zeros(shape=(m,m))
        
        inverse_fouriered = np.fft.ifft2(np.round(fouriered))
        st.image([to_img(black_and_white), Image.fromarray(np.uint8(np.round(inverse_fouriered)))],width=500)
        
        
        

    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    args = parser.parse_args()
    main(args)