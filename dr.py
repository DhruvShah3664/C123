import cv2
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#setting an https context to fetch data from OpenML
if(not os.environ.get("PYTHONHTTPSVERIFY", '')and getattr(ssl, "_create_unverifyed_context", None)):
    ssl._create_default_https_context = ssl._create_unverifyed_context

#fetching the data
X, y = fetch_openml("minst_784", version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,  train_size = 7500, test_size = 2500, random_state = 9)
XtrainScaled = Xtrain/255.0
XtestScaled = Xtest/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(XtrainScaled, ytrain)

ypred = clf.predict(XtestScaled)
accuracy = accuracy_score(ytest, ypred)
print(accuracy)

#starting the camera
cap = cv2.VideoCapture(0)
while(True):
    #Capture frame by frame
    try:
        ret, frame = cap.read()
        #drawing a box in the centre of the video
        height, width = gray.shape
        upperleft = (int(width/2-56), int(height/2-56))
        bottomright = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upperleft, bottomright, (0, 255, 0), 2)
        #to only consider the area inside the box for detecting the digit
        #roi = region of interest
        roi = gray[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]

        #converting cv2 image to pil format
        im_pil = Image.fromarray(roi)
        #convert to gray scale image-"L"fromat means each pixel is represented by a single value from 0 to 255
        im_bw = im_pil.convert("L")
        im_bw_resized = im_bw.im_bw_resized((28, 28), Image.ANTIALIAS)

        #inverting the image 
        im_bw_resized_inverted = PIL.ImageOps.invert(im_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(im_bw_resized_inverted, pixel_filter)
        im_bw_resized_inverted_scaled = np.click(im_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(im_bw_resized_inverted)
        im_bw_resized_inverted_scaled = np.asarray(im_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(im_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ", test_pred)

        cv2.imshow("frame", gray)
        if cv2.waitKey(1)&0XFF == ord("q"):
            break

    except Exception as e:
        pass
    
    cap.release()
    cv2.destroyAllWindows()