import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from numpy import argmax
import pandas as pd

# data loading
image_dir='C:/Users/Lenovo/Desktop/DL_assesment/cropped'
lionel_messi_images=os.listdir(image_dir+ '/lionel_messi')
maria_sharapova_images=os.listdir(image_dir+ '/maria_sharapova')
roger_federer_images=os.listdir(image_dir+ '/roger_federer')
serena_williams_images=os.listdir(image_dir+ '/serena_williams')
virat_kohli_images=os.listdir(image_dir+ '/virat_kohli')

print('The length of lional_images is',len(lionel_messi_images))
print('The length of maria_sharapova_images is',len(maria_sharapova_images))
print('The length of roger_federer_images is',len(roger_federer_images))
print('The length of serena_williams_images is',len(serena_williams_images))
print('The length of virat_kohli_images is',len(virat_kohli_images))


dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(lionel_messi_images),desc="Lional Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)

for i ,image_name in tqdm(enumerate(maria_sharapova_images),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

for i ,image_name in tqdm(enumerate(roger_federer_images),desc="roger_federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)    

for i ,image_name in tqdm(enumerate(serena_williams_images),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)


for i ,image_name in tqdm(enumerate(virat_kohli_images),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)   

dataset=np.array(dataset)
label = np.array(label)

print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))

print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.3,random_state=115)
print("Normalaising the Dataset.")
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape= (128,128,3)))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics= ['accuracy'])


print("Training Started.\n")
history = model.fit(x_train, y_train, epochs=20, batch_size = 64, validation_split = 0.2)
print("Training Finished.\n")

model.summary()

# Plot and save accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('C:/Users/Lenovo/Desktop/DL_assesment/cropped/results/cnn_accuracy_plot.png')

# Clear the previous plot

# Plot and save loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('C:/Users/Lenovo/Desktop/DL_assesment/cropped/results/cnn_loss_plot.png')


print("Model Evalutaion Phase.\n")
loss,accuracy= model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy*100,2)}')


results = model.predict(x_test)
results = argmax(results,axis = 1)
results = pd.Series(results,name="Predicted Label")
submission = pd.concat([pd.Series(y_test,name = "Actual Label"),results],axis = 1)
print(submission)
submission.to_csv("C:/Users/Lenovo/Desktop/DL_assesment/cropped/results/CNN1_face.csv",index=False)
print("--------------------------------------\n")