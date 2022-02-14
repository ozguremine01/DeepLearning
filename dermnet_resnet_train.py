import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import mod
from numpy.lib.type_check import imag
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.keras import initializers
import os
from keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.constraints import MinMaxNorm
from tensorflow.python.keras.models import Sequential

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.resnet50 import ResNet50

from keras.models import Model, load_model
import keras
from keras.layers import Flatten,Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

from keras.utils.vis_utils import plot_model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras_applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.optimizers import Adam


train='D:\\YL_derin_ogrenme_dersi_proje_cilt\\archive\\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration\\Train'
test='D:\\YL_derin_ogrenme_dersi_proje_cilt\\archive\\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration\\Test'

print(train)

train_datagen= ImageDataGenerator(rescale=1./255)
test_datagen= ImageDataGenerator(rescale=1./255)
train_generator= train_datagen.flow_from_directory(train,target_size=(150,150), batch_size=16, class_mode="categorical",color_mode='rgb')
test_generator=test_datagen.flow_from_directory(test,target_size=(150,150), batch_size=16, class_mode="categorical",color_mode='rgb')
"""
print(np.array(train_generator.class_indices))
print(np.array(train_generator.classes))
print(np.array(train_generator.image_shape))
print(np.array(test_generator.labels))
print(np.array(train_generator.color_mode))
print(train_generator)

print(list(train_generator.class_indices.items())[0][0])

print(list(test_generator.class_indices.items())[0][1])

print(len(list(train_generator.class_indices.items())))

print(len(list(train_generator.classes)))
print(len(list(test_generator.classes)))

print(train_generator.image_shape)

pre_trained=InceptionV3(input_shape=(150,150,3),include_top=False, weights=None)
#ResNet-50
add_model = Sequential()
add_model.add(pre_trained)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(9,activation='softmax'))

model_inceptionv3 = add_model
model_inceptionv3.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(),metrics=['accuracy'])
model_inceptionv3.summary()

history = model_inceptionv3.fit_generator(train_generator, 
                              steps_per_epoch=10, 
                              epochs=100, 
                              verbose=1)
model_inceptionv3.save('D:\\YL_derin_ogrenme_dersi_proje_cilt\\archive\\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration\\inceptionv3_100_epoch.h5')
"""
train_dir='D:\\YL_derin_ogrenme_dersi_proje_cilt\\archive\\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration\\Train\\'
test_dir='D:\\YL_derin_ogrenme_dersi_proje_cilt\\archive\\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration\\Test\\'
"""
from tensorflow.keras.optimizers import RMSprop 
import tensorflow as tf

train_data=ImageDataGenerator(rescale=1/255)
test_data=ImageDataGenerator(rescale=1/255)
train_dataset=train_data.flow_from_directory(train_dir, target_size=(150,150), batch_size = 10, class_mode= 'categorical')
print(train_dataset.class_indices)
print(train_dataset.classes)
test_dataset=test_data.flow_from_directory(test_dir, target_size=(150,150), batch_size = 10, class_mode= 'binary')
print(train_dataset.class_indices)
print(train_dataset.classes)
"""
categories= ['actinic_keratosis', 'basal_cell_carcinoma','dermatofibroma','melanoma','nevus','pigmented_benign_keratosis','seborrheic_keratosis','squamous_cell_carcinoma','vascular_lesion']

train_data=[]
test_data=[]
#import pickle

for category in categories:
    path=os.path.join(train_dir,category)
    label= categories.index(category)
    for img6 in os.listdir(path):
        imgpath=os.path.join(path,img6)
        image1=cv.imread(imgpath,0)
        
        img_array=np.array(image1).flatten()
        train_data.append([img_array,label])
print(len(train_data))

for category_test in categories:
    path=os.path.join(test_dir,category_test)
    label= categories.index(category_test)
    for img1 in os.listdir(path):
        imgpath=os.path.join(path,img1)
        image1=cv.imread(imgpath,0)
        
        img_array=np.array(image1).flatten()
        test_data.append([img_array,label])
print(len(test_data))


train_features=[]
train_labels=[]

test_features=[]
test_labels=[]
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
for t_feature,t_label in train_data:
    train_features.append(t_feature)
    train_labels.append(np.array(t_label))
for t_feature,t_label in test_data:
    test_features.append(t_feature)
    test_labels.append(np.array(t_label))

train_features=np.array(train_features)/255
test_features=np.array(test_features)/255
train_features.reshape(-1, 150, 150, 1)
train_labels = np.array(train_labels)
test_features.reshape(-1, 150, 150, 1)
test_labels = np.array(test_labels)

model_load= load_model('D:\\YL_derin_ogrenme_dersi_proje_cilt\\archive\\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration\\benim_modelim_100_epoch.h5')

predictions = model_load.predict_classes(test_features)
predictions = predictions.reshape(1,-1)[0]
from sklearn.metrics import classification_report
print(classification_report(test_labels,predictions,target_names=categories))

train_data=ImageDataGenerator(rescale=1/255)
test_data=ImageDataGenerator(rescale=1/255)

train_dataset=train_data.flow_from_directory(train_dir, target_size=(150,150), batch_size = 10, class_mode= 'categorical')
test_dataset=test_data.flow_from_directory(test_dir, target_size=(150,150,), batch_size = 10, class_mode= 'categorical')

"""
#modelim
model= tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(256,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])
"""
"""
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
model_fit=model.fit(train_dataset,epochs=100)

model.save('D:\\YL_derin_ogrenme_dersi_proje_cilt\\archive\\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration\\benim_modelim_100_epoch.h5')

acc = model_fit.history['accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')

plt.title('Training accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.show()
"""

"""
preds=model.predict(np.array(test_dataset.classes))
model.evaluate(test_features, test_labels)
"""
"""
test_loss,test_acc=model.evaluate(test_dataset,verbose=2)
print(test_acc)
"""
"""
deneme=np.round(model.predict(test_dataset))

print(deneme)
"""

#x_train, x_test, y_train, y_test=train_features,test_features,train_labels, test_labels

#x_train, x_test, y_train, y_test=train_test_split(train_features,train_labels, test_size=0.3)
"""
from sklearn.metrics import classification_report
predictions=model_load.predict_classes(x_test)
predictions=predictions.reshape(1,150,150,-1)[0]
print(classification_report(y_test, predictions, target_names=categories))
"""
"""
print(x_train)
print(y_train)
print(x_test)
print(y_test)


print(classification_report(y_test, np.around(model_load.predict(x_test))))
"""