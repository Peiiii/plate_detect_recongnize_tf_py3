
from detect.predict import Yolo_test
from recongnize.predict import Recongizer
import cv2,glob,os
import numpy as np

class Predictor:
    def __init__(self):
        self.D = Yolo_test()

        self.R = Recongizer()
        print('model loaded.')
    def predict_from_file(self,fp):
        x=self.D.predict_from_file(fp)
        # print(img)
        print('detect image succeeded.')
        #
        x=self.R.predict(x)
        # res=self.R.predict(np.array([cv2.imread('demo/2.jpg')]))
        return x
    def predict_batch(self,file_names):
        ys=[]
        for f in file_names:
            y=self.predict_from_file(f)
            ys.append(y)
        return ys

if __name__=='__main__':
    P=Predictor()
    y=P.predict_from_file('data/demo/6.jpg')
    # fs=glob.glob('data/demo/5.jpg')
    # ys=P.predict_batch(fs)
    # print(ys)
    print(y)



