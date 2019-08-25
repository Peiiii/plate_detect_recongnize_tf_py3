"""

"""

import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from . import cnn_lstm_otc_ocr
from .  import utils


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


import cv2
import itertools, os, time,glob
import numpy as np


# class ListIterator:
#     def __init__(self,mylist,batch_size):
#         self.list=mylist
#         self.batch_size=batch_size
#         self.length=len(mylist)
#         self.num_batch=self.length//self.batch_size
#         self.cur_batch_idx=-1
#         self.cur_batch=None
#     def next(self):
#         self.cur_batch_idx+=1
#         if self.cur_batch_idx>=self.num_batch:
#             self.cur_batch_idx=0
#         self.cur_batch=self.get_batch_by_idx(self.cur_batch_idx)
#         return self.cur_batch
#
#     def get_batch_by_idx(self,batch_idx):
#         st=batch_idx*self.batch_size
#         ed=st+self.batch_size
#         return self.list[st:ed]
#
#
#
#
#
# def loadInput(glob_str,imread_mode=cv2.IMREAD_COLOR,target_size=(272,72)):
#     # file_names=[input_dir+'/'+f for f in os.listdir(input_dir)]
#     file_names=glob.glob(glob_str)
#     print(file_names)
#     imgs=[cv2.imread(f,imread_mode).astype(np.float32) for f in file_names]
#     imgs=np.array([cv2.resize(img,target_size) for img in imgs])
#     imgs=(imgs/255)*2-1
#     return imgs,file_names
#
# def loadXY(input_dir,imread_mode=cv2.IMREAD_COLOR,target_size=(272,72),with_file_names=True):
#     def getLabel(fn):
#         label=os.path.basename(fn).split('_')[-1][:7]
#         return label
#     file_names=[input_dir+'/'+f for f in os.listdir(input_dir)]
#     labels=[getLabel(fn) for fn in file_names]
#     imgs=[cv2.imread(f,imread_mode).astype(np.float32) for f in file_names]
#     imgs=np.array([cv2.resize(img,target_size) for img in imgs])
#     imgs=(imgs/255)*2-1
#     return imgs,labels,file_names

class Recongizer:
    def __init__(self):
        self.imread_mode = cv2.IMREAD_COLOR
        self.img_size=(272,72)

        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        with self.graph.as_default():
            self.model = cnn_lstm_otc_ocr.LSTMOCR('infer')
            self.model.build_graph()
            self.saver=tf.train.Saver()
            # 注意！恢复器必须要在新创建的图里面生成,否则会出错。
        self.sess = tf.Session(graph=self.graph)  # 创建新的sess


        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                print('loading model  %s ...'%(ckpt))
                self.saver.restore(self.sess, ckpt)  # 从恢

    def predict(self,img):
        # print(img)
        print(img.shape)
        img=(cv2.resize(img,self.img_size)/255)*2-1
        img=np.array([img])
        y=self._predict(img)[0]
        return y

    def _predict(self,xs):
        feed = {self.model.inputs: xs}
        ys = self.sess.run(self.model.dense_decoded, feed)
        ys=self.decodePreds(ys)
        return ys

    def decodePreds(self,ys, verbose=False):
        letters2 = utils.charset

        def tensor_to_text(y):
            text = [letters2[i] for i in y]
            text = ''.join(text)
            return text

        if verbose:
            print('indexes:', ys)
        ys = [tensor_to_text(y) for y in ys]
        return ys

    #
    # def predict_from_file(self,fp):
    #     img=self.loadImage(fp)
    #     img=np.asarray([img])
    #     result=self._predict(img)[0]
    #     return result
    #
    # def loadImage(self,fp):
    #     img=cv2.imread(fp, self.imread_mode).astype(np.float32)
    #     img=cv2.resize(img,self.img_size)
    #     img = (img / 255) * 2 - 1
    #     return img


    # def predict_dir(self,dir):
    #     file_names = glob.glob(dir + '/*.jpg')
    #     xs = self.loadImgsBatch(file_names)
    #     ys = self.predict(xs)
    #     ys = self.decodePreds(ys)
    #     # showResult(labels,ys)
    #     self.saveResult('results/result.txt', file_names, ys)

    # def saveResult(self,file, labels, ys):
    #     s = ''
    #     for l, y in zip(labels, ys):
    #         s += '%s , %s\n' % (l, y)
    #     with open(file, 'w') as f:
    #         f.write(s)
    # def loadImgsBatch(self,file_names, imread_mode=cv2.IMREAD_COLOR, target_size=(272, 72)):
    #     imgs = [cv2.imread(f, imread_mode).astype(np.float32) for f in file_names]
    #     imgs = np.array([cv2.resize(img, target_size) for img in imgs])
    #     imgs = (imgs / 255) * 2 - 1
    #     return imgs





if __name__=="__main__":
    # xs,labels,file_names=loadXY('imgs/infer')
    # file_names=glob.glob('/home/user/datasets/results/ccpd/ccpd_chanllenge_results/*/*.jpg')
    # predict_test()
    # predict_dir('demo_imgs')
    pass