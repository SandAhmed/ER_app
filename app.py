import streamlit as st
from PIL import Image
import numpy as np
from models import EmotionModel

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@st.cache_resource
def load_model():
    return EmotionModel('models/emotion_model.h5')

model = load_model()


use_vgg = st.checkbox('Use VGG Model? ')

uploaded = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, width = 300)

    if st.checkbox('Predict'):
        model_type = 'vgg' if use_vgg else 'cnn'
        predictions = model.predict(image, model_type)
        top_idx = np.argmax(predictions)
        top_emotion = EMOTIONS[top_idx]
        top_confidence = predictions[top_idx]

        st.success(f"**Emotion:** {top_emotion.upper()}")
        st.info(f"**Confidence:** {top_confidence:.1%}")

        st.write("---")
        st.write("All predictions:")
        for i, (emotion, score) in enumerate(zip(EMOTIONS, predictions)):
            st.write(f"{emotion}: {score:.1%}")