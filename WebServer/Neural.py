import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import model_from_json


class NeuralNetwork:
    def __init__(self):
        with open("../Final Model/model1.json", "r") as f:
            self.model1 = model_from_json(f.read())
            f.close()

        self.model1.load_weights("../Final Model/modelOpt1.h5")

        with open("../Final Model/model2.json", "r") as f:
            self.model2 = model_from_json(f.read())
            f.close()

        self.model2.load_weights("../Final Model/modelOpt2.h5")

        with open("../Final Model/model3.json", "r") as f:
            self.model3 = model_from_json(f.read())
            f.close()

        self.model3.load_weights("../Final Model/modelOpt3.h5")

        print(self.model1.summary())
        print(self.model2.summary())
        print(self.model3.summary())

    def prediction(self, image):
        # data = cv2.imread("./static/image.jpg")
        data = image
        data = np.resize(data, (224, 224, 3))[..., ::-1].astype(float) / 255.0
        data = np.expand_dims(data, axis=0)
        # print(data.shape)

        pre1 = self.model1.predict(data).tolist()[0]
        pre2 = self.model2.predict(data).tolist()[0]
        pre3 = self.model3.predict(data).tolist()[0]
        pre = [(1*pre1[0]+6*pre2[0]+4*pre3[0])/11, (1*pre1[1]+6*pre2[1]+4*pre3[1])/11]

        res = [round(pre1[1]*100, 2), round(pre2[1]*100, 2), round(pre3[1]*100, 2), round(pre[1]*100, 2),
               "Oral Squamous Cell Carcinoma" if pre[0] <= pre[1] else "Normal Oral Cavity"]
        # print(res)
        return res

    def model_info(self, n):
        print(n)
        if n == '1':
            return self.model1.summary()
        elif n == '2':
            return self.model2.summary()
        elif n == '3':
            return self.model3.summary()
        else:
            return "Only 1,2,3 Valid Argument"
