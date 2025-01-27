import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array  
import matplotlib.pyplot as plt

model = load_model("model.h5")  

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

def detect_and_display(img_path, model, image_size=128):
    """
    Function to detect tumor and display results.
    If no tumor is detected, it displays "No Tumor".
    Otherwise, it shows the predicted tumor class and confidence.
    """
    try:
        img = load_img(img_path, target_size=(image_size, image_size))
        img_array = img_to_array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)  

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        if class_labels[predicted_class_index] == 'notumor':
            result = "No Tumor"
        else:
            result = f"Tumor: {class_labels[predicted_class_index]}"

        img_to_show = load_img(img_path)
        plt.imshow(img_to_show)
        plt.axis('off')
        plt.title(f"{result} (Confidence: {confidence_score * 100:.2f}%)")
        st.pyplot(plt) 

    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")


def main():
    st.title("Brain Tumor Detection System")
    st.markdown("Upload an image of a brain scan to check for the presence of a tumor.")
    

    uploaded_image = st.file_uploader("Upload Brain Scan Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:

        img_path = f"temp_{uploaded_image.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)  
        detect_and_display(img_path, model)

if __name__ == "__main__":
    main()
