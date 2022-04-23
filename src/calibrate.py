import streamlit as st
import numpy as np
from PIL import Image

if 'count' not in st.session_state:
    st.session_state.files = []

st.title('calibrate')

## upload files
uploaded_files = st.file_uploader("Upload image files", accept_multiple_files=True)

for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.session_state.files += [uploaded_file.name  + ' ' + str(img_array.shape)]

st.write("filenames:", st.session_state.files)