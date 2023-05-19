import numpy as np
import time
import streamlit as st
from keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils.image_utils import img_to_array

# Load model (absolute path)
model = load_model("D:\AI\VScode\model\model_Corn_Disease.h5")

def main():
    # Title the web
    st.title(":green[CORN'S LEAF DISEASE]")

    # Create menu tab
    tabs = ["HOME", "MORE"]
    active_tab = st.sidebar.radio("Select tab", tabs)

    # Display the content (depend on the following tab)
    if active_tab == "HOME":
        tool_tab()
    elif active_tab == "MORE":
        more_tab()

def tool_tab():
    # Display the header and "Drag and drop" bar
    st.header("DISEASE DETECTION AND DIAGNOSIS ON CORN PLANTS")
    st.write(":warning: Note about the image:") 
    st.write(":heavy_check_mark: _SHOULD be smooth image_")
    st.write(":heavy_check_mark: _MUST show the direct object_")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png"])
    
    # Process the file (if true)
    if uploaded_file is not None:
        successMessage = st.empty()
        successMessage = st.success("Upload image successfully!", icon="✅")
        # Create button
        click = st.button("Click here to detect right now!")
        img_PIL = Image.open(uploaded_file)
        img_array = np.array(img_PIL) # np.array -> RGB

        if(click):
            successMessage.empty()
            # status element: spinner
            with st.spinner('Loading...'):
                time.sleep(2)
            st.success('Done!')

            # Resize the image to the desired dimensions
            resized_img = cv2.resize(img_array, (40, 40))

            photo = img_to_array(resized_img)
            photo = photo.astype('float32')
            photo = photo / 255
            photo = np.expand_dims(photo, axis=0)

            result = model.predict(photo).argmax()

            class_name = ['Not Corn leaf', 'Healthy', 'CommonRust', 'LeafBlight']
            st.write(class_name[result])
            plt.imshow(img_array)

            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

def more_tab():
    st.header("Check my Git for more")
    st.markdown(":black[Link is here](https://github.com/LeMinhTruong16/AI)")

if __name__ == "__main__":
    main()
