import os
import streamlit as st
import cv2 as cv
import tempfile
from fashioniq import Vton
from PIL import Image

def apply_cloth(source_img, dest_img):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as source_temp, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as dest_temp:
            source_temp.write(source_img.read())
            dest_temp.write(dest_img.read())

            source_temp_path = source_temp.name
            dest_temp_path = dest_temp.name

        vton = Vton(source_temp_path, dest_temp_path)
        final_img, _ = vton.apply_cloth()
        return final_img
    except Exception as e:
        st.error(f"Error applying cloth: {e}")
        return None

def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(to top right, #ADD8E6, #90EE90);
        }
        .file-upload-btn {
            background-color: blue;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            cursor: pointer;
        }
        button#apply-cloth-btn {
            background-color: green !important;
            color: white !important;
            padding: 8px 12px !important;
            border-radius: 4px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            border: none !important;
            cursor: pointer !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Fashion IQ')
    
    uploaded_file1 = st.file_uploader("Upload Reference  Image", type=["jpg", "jpeg", "png", "webp"])
    uploaded_file2 = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(uploaded_file1, caption='Reference Image', width=150)
        with col2:
            st.image(uploaded_file2, caption='Profile Image', width=150)
        if st.button('Apply Cloth', key='apply-cloth-btn'):
            final_img = apply_cloth(uploaded_file1, uploaded_file2)
            if final_img is not None:
                final_img_pil = Image.fromarray(cv.cvtColor(final_img, cv.COLOR_BGR2RGB))
                with col3:
                    st.image(final_img_pil, caption='Output Image', width=150)

if __name__ == "__main__":
    main()







