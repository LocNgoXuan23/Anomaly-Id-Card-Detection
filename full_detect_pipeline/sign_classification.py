import numpy as np
from dataset import img_transformer
import cv2

def sign_classifier(img, model):
    result = model(np.expand_dims(img_transformer(img), axis=0))
    c = np.argmax(result, axis=-1)[0]
    acc = round(float(max(result[0])), 3)
    img = cv2.putText(
        img, f'{c}-{acc}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FONT_HERSHEY_SIMPLEX, 2, cv2.LINE_AA
        )

    return img, c