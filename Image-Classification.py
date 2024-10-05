#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model


# In[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential


# In[3]:


train_path='E:\Hoc\clone_github\Real-Time-Image-Classification\DataSet\Training'
test_path='E:\Hoc\clone_github\Real-Time-Image-Classification\DataSet\Testing'


# In[4]:


x_train=ImageDataGenerator(rescale=1./255).flow_from_directory(train_path,batch_size=32,classes=['female','male'],class_mode='categorical',target_size=(220,220),shuffle=True)
x_test=ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,batch_size=32,classes=['female','male'],class_mode='categorical',target_size=(220,220),shuffle=True)


# In[6]:


from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential


# In[6]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(220 , 220 , 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax')) 
tf.keras.utils.plot_model(model, show_shapes=True)


# In[7]:


model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history=model.fit(x_train, epochs=50,steps_per_epoch=(2750//55),verbose=1,validation_data=x_test,validation_steps=(600//60))


# In[10]:


model.save('E:\Hoc\clone_github\Real-Time-Image-Classification\Saved_Model.keras')


# In[8]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc='best')


# In[9]:


face_classifier=cv2.CascadeClassifier('E:\Hoc\clone_github\Real-Time-Image-Classification\haarcascade_frontalface_default.xml')


# In[10]:


model=load_model('E:\Hoc\clone_github\Real-Time-Image-Classification\Saved_Model.keras')


# In[11]:


def predict(model,cap):
    prediction=model.predict(cap)
    return np.argmax(prediction)


# In[12]:


vid=cv2.VideoCapture(0)
if not vid.isOpened():
    print('cannot open the camera')
    exit()
while True:
    ret,cap=vid.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray=cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
    face=face_classifier.detectMultiScale(gray, 1.05, 5)
    if face is ():
        print('no face detected')
        
    for (x,y,a,b) in face:
        cv2.rectangle(cap,(x,y),(x+a,y+b),(127,0,255),2)
        roi_color=cap[y:y+b,x:x+a]
        img = cv2.resize(roi_color, (220,220), interpolation=cv2.INTER_CUBIC)
        img = img.astype('float16')
        img = np.expand_dims(img, axis=0)
        score=predict(model,img)
        if score==1:
            print('Male')
            cv2.putText(cap,'Male',(x+a,y+b), cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255,0,0),1, cv2.LINE_AA)
        if score==0:
            print('Female')
            cv2.putText(cap,'Female',(x+a,y+b), cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255,0,0),1, cv2.LINE_AA)  
        cv2.imshow('image',cap)
        cv2.waitKey(2)
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
    


# In[13]:


vid.release()
cv2.destroyAllWindows()


# In[ ]:




