import tensorflow as tf
import numpy as np

class AI:
    def __init__(self):
        self.model = tf.keras.models.load_model('model/snakeMove.h5')

    def predict(self, x_snake, y_snake, x_apple, y_apple, x_sub, y_sub):
        x = np.array([[x_snake, y_snake, x_apple, y_apple, x_sub, y_sub]])
        result = self.model.predict(x)
        direction = np.argmax(result)
        return direction