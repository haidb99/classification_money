import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Flatten, Input, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


def model_tranferLN():
    # Tải weight của mạng VGG16 với bộ dữ liệu imagenet
    conv_vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(128, 128, 3)))

    # Đóng băng các conv_layer của VGG16
    for layer in conv_vgg16.layers:
        layer.trainable = False

    # Lấy output của mạng Vgg16 sau khi bỏ qua FC
    fcHead = conv_vgg16.output
    # Thêm các lớp FC
    fcHead = Flatten()(fcHead)
    fcHead = Dense(4096, activation = "relu", name = 'fc1')(fcHead)
    fcHead = Dropout(0.5)(fcHead)
    fcHead = Dense(4096, activation = "relu", name = 'fc2')(fcHead)
    fcHead = Dropout(0.5)(fcHead)
    fcHead = Dense(7, activation = "softmax", name = 'fc3')(fcHead)
    my_model = Model(inputs=conv_vgg16.input, outputs=fcHead)
    return my_model

def run_model():
    # Đọc dữ liệu
    df = pd.read_csv('data.csv')
    df.head()
    y = df['49152'].values
    X = df.drop(columns=['49152'], axis=1)
    X = X.values
    X = X.reshape(X.shape[0], 128, 128, 3)

    # chuẩn hóa label
    y[y == 10000] = 1
    y[y == 20000] = 2
    y[y == 50000] = 3
    y[y == 100000] = 4
    y[y == 200000] = 5
    y[y == 500000] = 6

    # Chia dữ liệu train, val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

    # One hot endcoding
    Y_train = np_utils.to_categorical(y_train, 7)
    Y_val = np_utils.to_categorical(y_val, 7)

    # Khởi tạo model
    my_model = model_tranferLN()
    # compile
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # augmentation,
    # rescale = phóng to, thu nhỏ
    # Rotation_range: xoay
    # height_shift_range, width_shift_range: dịch trái phải, trên xuống
    # brightness_range: chỉnh độ sáng
    # horizontal_flip: Lật ảnh
    # zoom_range: phóng to
    aug_train = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   brightness_range=[0.2, 1.5],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   zoom_range=0.1
                                   )
    aug_val = ImageDataGenerator(rescale=1. / 255)
    my_model.summary()
    #Checkpoint
    checkpoint = ModelCheckpoint('models/model.h5',
                                 save_best_only=True, monitor='val_loss', verbose=0)

    #train model
    batch_size = 32
    np_epochs = 2
    steps_epoch = X_train.shape[0]//batch_size
    val_steps = X_val.shape[0]//batch_size

    # load weights
    my_model.load_weights('models/model.h5')
    my_model.fit_generator(aug_train.flow(X_train, Y_train, batch_size = batch_size),
                           validation_data=aug_val.flow(X_val, Y_val, batch_size = batch_size),
                           epochs=np_epochs,
                           callbacks = [checkpoint],
                           steps_per_epoch = steps_epoch,
                           validation_steps = val_steps,
                           verbose = 1)
    print("done!")

#run_model()