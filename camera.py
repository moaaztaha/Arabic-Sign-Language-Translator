# -*- coding: UTF-8 -*-
import cv2
import datetime
from Classifier import *
from translate import translate

x = 10
y = 60
xs = 250
ys = 300
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):

    char = ''
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()
    def getChar(self):
        return self.char
        
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        cv2.rectangle(image, (x, y), (xs, ys), (0, 255, 0), 3)
        
        img_name = "opencv_frame_{}.png".format(
            datetime.datetime.now().microsecond)
        # cv2.imwrite(img_name, image)
        # print("{} written!".format(img_name))

        # img = cv2.imread(img_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        crop_img = gray[y:ys, x:xs]
        #cv2.imwrite(img_name, crop_img)
        ## Classsifer
        self.char = classify(crop_img)
        cv2.putText(image,self.char,(x,y), font, 2, (255,255,255), 2, cv2.LINE_AA)
        
        print(self.char)
        ret, jpeg = cv2.imencode('.jpg', crop_img)

        return jpeg.tobytes()

    
    