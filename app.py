import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(layout='wide')
st.title("PCB Defect Detection")

# Load YOLO model
model = YOLO("best.pt")

st.write("### Upload an image for prediction")
file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if file is not None:
    # Display uploaded image
    img = Image.open(file)
    st.image(img, caption="Uploaded Image")
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Run YOLO prediction (this will create a new predict folder)
    results = model.predict(img_array, imgsz=640, conf=0.3, save=True)
    
    st.write("### Prediction Results")
    
    # Find the latest "predict*" folder in runs/detect
    detect_folder = "runs/detect"
    predict_folders = [f for f in os.listdir(detect_folder) if f.startswith("predict")]
    
    if predict_folders:
        # Sort folders by creation time and get the latest
        latest_folder = max(predict_folders, key=lambda f: os.path.getmtime(os.path.join(detect_folder, f)))
        latest_folder_path = os.path.join(detect_folder, latest_folder)
        
        # Get the latest image in that folder
        pred_images = sorted(
            os.listdir(latest_folder_path),
            key=lambda x: os.path.getmtime(os.path.join(latest_folder_path, x))
        )
        
        if pred_images:
            pred_image_path = os.path.join(latest_folder_path, pred_images[-1])
            st.image(pred_image_path, caption="Predicted Image")
        else:
            st.write("No images found in the latest predict folder.")
    else:
        st.write("No predict folders found.")
