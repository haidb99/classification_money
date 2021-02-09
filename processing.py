import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import os
import time


data_folder = os.getcwd()
data_folder = os.path.join(data_folder, 'data')

# show ảnh
def show_img(img):
    img = img.astype(np.uint8)
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Thêm khung cho ảnh
def region_img(img):
    # cạnh trên
    x1, x2 = ((58, 18), (582, 18))
    img2 = cv2.line(img, x1, x2, color=(0, 255, 0), thickness=2)

    # cạnh dưới
    x1, x2 = ((58, 302), (582, 302))
    img2 = cv2.line(img2, x1, x2, color=(0, 255, 0), thickness=2)

    # cạnh trái
    x1, x2 = ((58, 18), (58, 302))
    img2 = cv2.line(img2, x1, x2, color=(0, 255, 0), thickness=2)

    # cạnh phai
    x1, x2 = ((582, 18), (582, 302))
    img2 = cv2.line(img2, x1, x2, color=(0, 255, 0), thickness=2)
    return img2

# Tạo dữ liệu
def write_data(data_folder, label):
    # tạo thư mục chứa ảnh có tên là label
    try:
        os.chdir(data_folder)
        os.mkdir(label)
    except:
        pass

    # Lưu ảnh vào thư mục (300 ảnh)
    cap = cv2.VideoCapture(0)
    dem = 0
    while (dem <= 580):
        ret, frame = cap.read()
        frame = region_img(frame)
        cv2.imshow('', frame)
        if (dem > 80):
            img = frame[20:300, 60:580, :]
            name = str(dem - 80) + ".jpg"
            img_name = os.path.join(data_folder, label)
            img_name = os.path.join(img_name, name)
            cv2.imwrite(img_name, img)
            print(img_name)
            time.sleep(0.1)

        if(dem == 330):
            time.sleep(5)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        dem = dem + 1
    cap.release()
    cv2.destroyAllWindows()
    print("done!")

# Đọc dữ liệu và resize lại ảnh (128, 128), lưu vào file csv
def save_data():
    folder = 'data/'
    X = []
    y = []
    for moneyfolder in os.listdir(folder):
        if(moneyfolder != '.ipynb_checkpoints'):
            for file in os.listdir(folder + moneyfolder):
                img = cv2.imread(folder + moneyfolder + '/' + file)
                img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
                X.append(img)
                y.append(moneyfolder)
    X = np.array(X)
    y = np.array(y)
    df = DataFrame(X.reshape(X.shape[0], 128*128*3))
    df[49152] = y
    df.to_csv('data2.csv', index = None, header = True)
    print("Done!")
    return df

#write_data(data_folder, '500000')
# save_data()
# img = cv2.imread('data/test/99.jpg')
# show_img(img2)
# Region_img(img)


#...........................................
# img = cv2.imread('3.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img,(3,3),0)
# img = cv2.adaptiveThreshold(img, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,6)
#
# print(img.shape)
# show_img(img)
# cv2.imwrite('3-copy.jpg', img)