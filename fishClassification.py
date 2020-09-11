import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD,Adam
# from tensorflow.keras.applications import VGG16
# import pickle
from tensorflow.keras.losses import CosineSimilarity
from random_eraser import get_random_eraser
import datetime

## show results by tensorboard
log_dir = 'logs\\fit\\' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboar_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

epoches = 100

aug= ImageDataGenerator(rotation_range=30,
                        width_shift_range=.1,
                        height_shift_range=.1,
                        shear_range=.2,
                        zoom_range=.1,
                        horizontal_flip=True,
                        fill_mode='nearest'
                        # preprocessing_function=get_random_eraser(v_l=0, v_h=1)
                        )

data=[]
labels=[]
for i,item in enumerate(glob.glob(r'C:\Users\mohammad\Desktop\preprocessod images\*\*')):
    
    img=cv2.imread(item)
    r_img=cv2.resize(img,(128,128))

    data.append(r_img)

    label=item.split('\\')[-2].split('.')[0]
    # print(label)
    labels.append(label)

    if i%100==0:
        print('Notice {} of data processod'.format(i))
    # if i==200:
    #     break
    
data=np.array(data)/255

## onehot encoding
le= LabelEncoder()
labels=le.fit_transform(labels)

labels=to_categorical(labels,13)

## using class weight because of unbalancing
classTotals=labels.sum(axis=0)
classWeight=classTotals.max()/classTotals
classWeight={0:classWeight[0],1:classWeight[1],2:classWeight[2],3:classWeight[3],4:classWeight[4],5:classWeight[5],6:classWeight[6],
            7:classWeight[7],8:classWeight[8],9:classWeight[9],10:classWeight[10],11:classWeight[11],12:classWeight[12]}

## split data to train and test data
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=.2)



# with open(r'C:\Users\mohammad\Documents\python code\practise deeplearning\dog_cat.pkl','rb') as f:
#     X_train,X_test,y_train,y_test=pickle.load(f)


# baseModel= VGG16(
#                 weights='imagenet',
#                 include_top=False,
#                 input_tensor=layers.Input(shape=(32,32,3))
#                 )

# for layer in baseModel.layers:
#     layer.trainable=False

## bulid CNN from scratch

# Network= models.Sequential([
#                             layers.Conv2D(32,(8,8),activation='relu',input_shape=(32,32,3)),
#                             layers.BatchNormalization(),
#                             layers.MaxPool2D((2,2)),

#                             # layers.Conv2D(64,(3,3),activation='relu'),
#                             # layers.BatchNormalization(),
#                             # layers.MaxPool2D((2,2)),

#                             layers.Conv2D(64,(5,5),activation='relu'),
#                             # baseModel,
#                             layers.BatchNormalization(),
#                             layers.MaxPool2D((4,4)),
#                             layers.Flatten(),

#                             layers.Dense(128,activation='relu'),
#                             layers.Dropout(.5),
#                             layers.Dense(64,activation='relu'),
#                             layers.Dense(13,activation='softmax')
#                             ])
                      
## transfer learning
base_model = tf.keras.applications.MobileNet(input_tensor=layers.Input(shape=(128,128,3)),include_top=False)
base_model.trainable = False

Network = tf.keras.Sequential([
  base_model,
# using BatchNormalization
#   layers.BatchNormalization(),
  tf.keras.layers.GlobalAveragePooling2D(),
  layers.Dense(64,activation='relu'),
  # using dropOut
  layers.Dropout(.2),
  layers.Dense(32,activation='relu'),
  # using dropOut
  layers.Dropout(.2),
  tf.keras.layers.Dense(13, activation='softmax')
])

## using learning rae decay
opt= SGD(lr=0.01,decay=0.001/epoches)

Network.compile(optimizer=opt,
                # loss='binary_crossentropy',
                loss=CosineSimilarity(axis=1),
                metrics=['accuracy'])


H = Network.fit_generator(aug.flow(X_train,y_train,batch_size=32),
                        epochs=epoches,validation_data=(X_test,y_test),
                        steps_per_epoch=len(X_train)//32,class_weight=classWeight,callbacks=[tensorboar_callback])

loss,acc=Network.evaluate(X_test,y_test)
print('accuracy : {:.2f}'.format(acc))


print(Network.summary())
# Network.save('out_CNN.h5')

## Convert the model.
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

## Save the TF Lite model.
# with tf.io.gfile.GFile('model.tflite', 'wb') as f:
#   f.write(tflite_model)

plt.plot(np.arange(epoches),H.history['accuracy'],label='train_accuracy')
plt.plot(np.arange(epoches),H.history['val_accuracy'],label='test_accuracy')
plt.plot(np.arange(epoches),H.history['loss'],label='loss')
plt.plot(np.arange(epoches),H.history['val_loss'],label='val_loss')

plt.legend()
plt.xlabel='epoches'
plt.ylabel='accuracy/loss'
plt.show()
