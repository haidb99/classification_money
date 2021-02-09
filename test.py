import cv2
import numpy as np
from train import model_tranferLN
from processing import *

# load model
model = model_tranferLN()
model.load_weights('models/model2.h5')

def predict(frame):
    img = frame[20:300, 60:580, :]
    img = cv2.resize(frame, (128, 128), cv2.INTER_AREA)
    img = img.astype(np.float)
    img = img.reshape([1, 128, 128, 3])
    img = img*1/255
    rs = model.predict(img)
    label_stt = np.argmax(rs[0])
    print([label_stt, rs[0][label_stt]])
    if( (rs[0][label_stt] > 0.8) and (label_stt != 0) ):
        if(label_stt == 1):
            return '10000'
        if(label_stt == 2):
            return '20000'
        if(label_stt == 3):
            return '50000'
        if(label_stt == 4):
            return '100000'
        if(label_stt == 5):
            return '200000'
        if(label_stt == 6):
            return '500000'
    return '0'
def show_predict(img, label):
    if (label !=  '0'):
        # set vi tri
        ord = (50, 50)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cỡ chứ
        fontscala = 1

        # độ dày
        thickness = 2

        # Màu chữ trong BGR
        color = (255, 0, 0)

        # Nội dung:
        return cv2.putText(img, label, ord, font, fontscala, color, thickness, cv2.LINE_AA)
    return img
def run():
    # load webcam
    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        img = region_img(frame)
        label = predict(img)
        img = show_predict(img, label)
        cv2.imshow('', img)
        time.sleep(0.1)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

#show_img(img)
run()

#img = cv2.imread('data/000000/400.jpg')
#predict(img)

