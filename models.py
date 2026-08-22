import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

class EmotionModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model('models/emotion_model.h5', compile=False)

    def predict(self, image, model_type = 'cnn'):
        if isinstance(image, Image.Image):
            img = np.array(image)

        else:
            img = image

        if model_type == 'cnn':
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.resize(img, (75, 75))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=[0, -1])

        else:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32)

            from tensorflow.keras.applications.vgg16 import preprocess_input
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            predictions = self.model.predict(img, verbose=0)[0]
            return predictions