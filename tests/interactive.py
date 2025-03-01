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
    st.title('Compression of data')
    st.header('Inputs')
    uploaded_file = st.file_uploader("Upload an image")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img)
    second_uploaded_file = st.file_uploader("Upload another image")
    if second_uploaded_file is not None:
        img = Image.open(second_uploaded_file)
        st.image(img)

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        array = np.asarray(img)
        #st.data_editor(array[:,:,0])
        #st.data_editor(array[:,:,1])
        #st.data_editor(array[:,:,2])

        st.header("Transformations of the first image")
        black_and_white = np.round(0.3*array[:,:,0] + 0.6*array[:,:,1] + 0.11*array[:,:,2])
        #st.data_editor(black_and_white)
        st.image(Image.fromarray(np.uint8(black_and_white)),width=500)

        fouriered = np.fft.fft2(black_and_white)
        angles = np.angle(fouriered)
        #st.data_editor(np.uint8((angles)/2/np.pi*256))
        st.subheader("Normalized phases (-pi,pi) -> (0,255)")
        st.image(Image.fromarray(np.uint8((angles)/360*256)),width=500)
        magnitudes = np.abs(fouriered)
        st.subheader(f"Normalized frequency magnitudes (0,{np.max(magnitudes)})->(0,255)")
        #st.image(Image.fromarray(np.uint8(
        #    np.round((magnitudes/np.max(magnitudes)*255))
        #    )),width=500)
        st.image(Image.fromarray(np.uint8(
            np.log(magnitudes+1)*10
            )),width=500)
        st.header("Inverse fourier with the highest frequencies forgotten")
        st.text("Starts omitting values in a square from the right bottom corner")
        a = st.slider("Square size",0,fouriered.shape[0],fouriered.shape[0])
        st.text(f"{a}")
        m = a
        fouriered_copy = fouriered.copy()
        fouriered_copy[fouriered.shape[0]-m:fouriered.shape[0],fouriered.shape[1]-m:fouriered.shape[1]] = np.zeros(shape=(m,m))
        
        inverse_fouriered = np.fft.ifft2(np.round(fouriered_copy))
        st.image([to_img(black_and_white), Image.fromarray(np.uint8(np.round(inverse_fouriered)))],width=500)

        if second_uploaded_file is not None:
            st.header("Phases from the first image mixed with magnitudes from the second image")

            img2 = Image.open(second_uploaded_file)
            array2 = np.asarray(img2)
            black_and_white2 = np.round(0.3*array2[:,:,0] + 0.6*array2[:,:,1] + 0.11*array2[:,:,2])
            fouriered2 = np.fft.fft2(black_and_white2)
            radius = np.abs(fouriered2)
            angle_and_radius_mix = np.zeros(fouriered2.shape, dtype=complex)
            for x in np.arange(fouriered.shape[0]):
                for y in np.arange(fouriered.shape[1]):
                    angle_and_radius_mix[x,y] = np.cos(angles[x,y]) + np.sin(angles[x,y])*1j
                    angle_and_radius_mix[x,y] *= radius[x,y]
            inverse_fft_mix =  np.fft.ifft2(angle_and_radius_mix)
            st.subheader("Source images")
            st.image([to_img(black_and_white),to_img(black_and_white2)],width=500)
            st.subheader("Mixed result")
            st.image(to_img(inverse_fft_mix),width=500)
            st.subheader("Reference: Average of the two input images")
            st.image(to_img((black_and_white+black_and_white2)/2.),width=500)
            
        
        
        

    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    args = parser.parse_args()
    main(args)