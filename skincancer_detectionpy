import tensorflow as tf
from tensorflow.keras import  layers, models
import cv2 as cv
#from matplotlib import pyplot as plt 
import os,shutil
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
 
dir="C:\\Users\\Pc\\Downloads\\archive\\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration\\Train\\"
data_dir_list=os.listdir(dir)
print(data_dir_list)
"""
data_dir_list=os.listdir(dir)
print(data_dir_list)

labels=[]

for i in os.listdir(dir):
    labels.append(i)

print(labels)
veri=[]
training_data=[]
for category in labels:
    path=os.path.join(dir,category)
    class_num=labels.index(category)    
    veri.append(path)
    for img in os.listdir(path):
        try:
            img_array=cv.imread(os.path.join(path,img))
            new_array=img_array.reshape(-1,150,150,1)
            training_data.append([new_array,class_num])
        except Exception as e:
            pass
print(veri)
print(len(training_data))
print(os.listdir(veri[0]))
x=[]
y=[]

for feature,label in training_data:
    x.append(feature)
    y.append(label)
"""
labels=[]
veri=[]
training_data=[]
görsel=[]
x=[]
y=[]
covid=[]
normal=[]
data=[]
from keras.utils import to_categorical
l=[]


class train_data:   
    def __init__(self,path):
        self.path=path       
        for i in os.listdir(self.path):
            labels.append(i)
            veri.append(dir +i+"\\")
            print("deneme label: ",i)
            print(dir +i+"\\")
            print(len(os.listdir(dir +i+"\\")))
        print("****************************Category etiket haline getirildi***************************************")
        for category in labels:
            path=os.path.join(path,category)  
            class_num=labels.index(category)
            
            print("deneme class_num: ", class_num)   
            for img in os.listdir(veri[class_num]):
                try:
                    img_array= cv.imread(veri[class_num]+img,0)
                    new_array=cv.resize(img_array,(150,150))
                    covid.append([np.array(new_array),np.array(class_num)]) 
                    l.append(class_num)     
                except Exception as e:
                    pass
        print("********************Category ve görseller tek boyutlu matris halinde eşleştirilir******************************") 
        """
        for a in range(9):
            for img in os.listdir(veri[a]):
                try:
                    img_array= cv.imread(veri[a]+img,0)
                    new_array=cv.resize(img_array,(150,150))
                    covid.append([np.array(new_array), np.array(a)]) 
                            
                except Exception as e:
                    pass
        """
        
        """
        for img1 in os.listdir(veri[1]):
            try:
                img_array1= cv.imread(img1)
                new_array1=img_array1.reshape(-1,150,150,1)
                normal.append([new_array1,class_num])
                    
            except Exception as e:
                pass                 
        """
    @staticmethod
    def veri_goster(a):        
        #pass
        print(os.listdir(veri[a]))
        #print(covid)
        #print(normal)
        
 

yol=train_data(dir)
print("************************************************************************************************")
print(train_data.veri_goster(0))
print("************************************************************************************************")
#print(os.listdir(dir+"covid\\"))
#print(veri[1])
#print(covid[50])
#print(covid[50][1])
from sklearn.model_selection import train_test_split
features=[]
labels1=[]
for i in covid:
    features.append(np.array(i[0]))
    labels1.append(np.array(i[1]))

print(len(features))
print(len(labels1))
print(len(os.listdir(dir)))
print(l)

"""
datagen=ImageDataGenerator()
datagen.fit(features)
"""

#train_datagen=datagen.flow_from_directory(dir,class_mode='categorical',batch_size=64)
x_train, x_test, y_train, y_test = train_test_split(features,labels1, test_size=0.30, random_state=10)

#Önemli Model ilk modelim

model=models.Sequential()

model.add(layers.Conv2D(16,(3,3), activation='relu',input_shape=(150,150,1), padding='same'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32,(3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

#model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(9, activation='softmax'))

print(model.summary())

#model.fit_generator(train_datagen, steps_per_epoch=16)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train)

model.predict(x_test[4])

print(y_test[4])
