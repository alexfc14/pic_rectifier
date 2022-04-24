import streamlit as st
import numpy as np
from PIL import Image
from cv import calibrate, undistort

if 'count' not in st.session_state:
    st.session_state.calibration_images = []
    st.session_state.distorted_images = {}
    st.session_state.undistorted_images = {}
    st.params = None

st.title('calibrate')

## upload files calibration pics
calibration_files = st.file_uploader("Upload calibration images", accept_multiple_files=True)
for f in calibration_files:
    image = Image.open(f)
    img_array = np.array(image)
    
    st.session_state.calibration_images += [img_array]

# st.write("filenames:", st.session_state.files)

if calibration_files:
    st.text('start calibrating')
    params = calibrate(st.session_state.calibration_images)
    st.text('finished calibrating')
    st.session_state.params = params

    ## upload files calibration pics
    distorted_files = st.file_uploader("Upload images to undistort", accept_multiple_files=True)
    for f in distorted_files:
        image = Image.open(f)
        img_array = np.array(image)
        print('distorted image', f.name)
        st.session_state.distorted_images[f.name] = img_array
        st.image(img_array)

    for f in distorted_files:
        img_array = st.session_state.distorted_images[f.name]
        undistorted = undistort(img_array, *params)
        st.session_state.undistorted_images[f.name] = undistorted
        st.image(undistorted)
