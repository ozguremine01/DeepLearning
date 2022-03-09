from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from keras import optimizers
from keras.models import Model

train_data_dir1="D:/YL/CT/yeni_ct/train/"
valid_data_dir1="D:/YL/CT/veri/valid/"
test_data_dir1="D:/YL/CT/yeni_ct/test/"

train_datagen= ImageDataGenerator(rescale=1./255)
test_datagen= ImageDataGenerator(rescale=1./255)
train_generator= train_datagen.flow_from_directory('D:\\YL\\CT\\yeni_ct\\train\\',target_size=(150, 150), batch_size=20, class_mode="binary")
test_generator=test_datagen.flow_from_directory('D:\\YL\\CT\\yeni_ct\\test\\',target_size=(150,150), batch_size=20, class_mode="binary")

print(np.array(train_generator.class_indices))
print(np.array(train_generator.classes))
print(np.array(train_generator.image_shape))
print(np.array(train_generator.labels))
print(np.array(train_generator.color_mode))
print(train_generator)


categories=['covid', 'noncovid']

for category in categories:
    path= os.path.join(train_data_dir1, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img_array, cmap="gray")
        #plt.show()
        #break
    #break
print(img_array)
print(img_array.shape)
training_data=[]
def create_training_data():
    for category in categories:
        path= os.path.join(train_data_dir1, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            
                img_array=cv2.imread(os.path.join(path, img), cv2.THRESH_BINARY)
                
                print(img_array.shape)
                d2_train_dataset = img_array.reshape(22500,)
                training_data.append([d2_train_dataset, class_num])
   

create_training_data()
print(len(training_data))
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

x = []
y = []

for feature, labels in training_data:  
    x.append(feature)
    y.append(labels)

print(x[2])
print(y)
from sklearn.model_selection import train_test_split

#a , b = train_test_split(np.array(x),test_size=0.4)  
#print(a)
#print(b)
print(len(y))
print(len(x))

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
print(x_test)
print(y_test)
print(x_train)
print(y_test)

print(x_train[4].shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
model=LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(model.score(x_test,y_test))
img1=x_test[30]
input_img=np.expand_dims(img1, axis=0)
prediction = np.argmax(model.predict(input_img))
print(y_test[30])
print("SVM Prediction sonuç:", prediction)
#svm_pred= model.predict(x_test)
#print(svm_pred)
print(pred)
print(y_test)

import matplotlib.pyplot as plt







    


