
from os import listdir
from numpy import asarray, save
from keras.utils import load_img
from keras.utils import img_to_array
import matplotlib.pyplot as plt
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
%matplotlib inline


folder = '/content/drive/MyDrive/AI_Final/1010AnhDeTrain/'
photos, labels = list(), list()
for file in listdir(folder):
  output= 0.0
  if file.startswith('man_01'):
     output= 1.0
  if file.startswith('man_003'):
    output= 2.0
  if file.startswith('man_10'):
    output= 3.0
  if file.startswith('man_11'):
    output= 4.0
  if file.startswith('man_15'):
    output= 5.0
  if file.startswith('man_20'):
    output= 6.0
  if file.startswith('man_22'):
    output= 7.0
  if file.startswith('man_28'):
    output= 8.0
  if file.startswith('man_32'):
    output= 9.0
  if file.startswith('man_36'):
    output= 10.0
  if file.startswith('man_40'):
    output= 11.0
  if file.startswith('man_47'):
    output= 12.0
  if file.startswith('man_60'):
    output= 13.0
  if file.startswith('woman_002'):
    output= 14.0
  if file.startswith('woman_04'):
    output= 15.0
  if file.startswith('woman_10'):
    output= 16.0
  if file.startswith('woman_11'):
    output= 17.0
  if file.startswith('woman_14'):
    output= 18.0
  if file.startswith('woman_15'):
    output= 19.0
  if file.startswith('woman_21'):
    output= 20.0
  if file.startswith('woman_22'):
    output= 21.0
  if file.startswith('woman_23'):
    output= 22.0
  if file.startswith('woman_25'):
    output= 23.0
  if file.startswith('woman_32'):
    output= 24.0
  if file.startswith('woman_38'):
    output= 25.0

  photo = load_img(folder + file, target_size= (96,72))
  photo= img_to_array(photo)

  photos.append(photo)
  labels.append(output)
     

photos= asarray(photos)
labels= asarray(labels)
print(photos.shape, labels.shape)
save('/content/drive/MyDrive/AI_Final/ThuMucLuuTru/NhanDien_photos.npy', photos)
save('/content/drive/MyDrive/AI_Final/ThuMucLuuTru/NhanDien_labels.npy', labels)
     

     # Split data into train & test
split_index = int(0.05 * len(photos))
test_x, test_y = photos[:split_index], labels[:split_index]
train_x, train_y = photos[split_index:], labels[split_index:]
     

     
print(test_x.shape, train_x.shape)


train_x = train_x.reshape((960, 96, 72, 3))
train_x = train_x.astype('float32')/255

test_x = test_x.reshape((50, 96, 72, 3))
test_x = test_x.astype('float32')/255

from keras.utils import to_categorical
train_y = to_categorical(train_y,26)
test_y = to_categorical(test_y,26)

from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D,Normalization,Input
from keras.optimizers import Adam
from keras import losses
loss = losses
batch_size = 64
epochs = 100
classes = 26

from keras.layers import LeakyReLU
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = 'linear', input_shape = (96, 72, 3), padding= 'same'))
model.add(LeakyReLU(alpha = 0.1))

model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(64, (3,3), activation = 'linear', padding = 'same'))
model.add(Conv2D(64, (3,3), activation = 'linear', padding = 'same'))
model.add(LeakyReLU(alpha = 0.1))

model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(128, (3,3), activation = 'linear', padding = 'same'))
model.add(LeakyReLU(alpha = 0.1))

model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(256, (3,3), activation = 'linear', padding = 'same'))
model.add(LeakyReLU(alpha = 0.1))

model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(512, (3,3), activation = 'linear', padding = 'same'))
model.add(LeakyReLU(alpha = 0.1))

model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(1024, (3,3), activation = 'linear', padding = 'same'))
model.add(LeakyReLU(alpha = 0.1))

#Đưa vào ANN, bộ ANN để phân loại:
from keras.losses import categorical_crossentropy
model.add(Flatten())
model.add(Dense(1024, activation = 'linear'))

model.add(Dense(classes, activation = 'softmax'))
model.summary()

#Compile:
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
train = model.fit(train_x, train_y, batch_size= batch_size, epochs= epochs, verbose= 1)

# Plot accuracy
import matplotlib.pyplot as plt
plt.plot(train.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Plot loss
plt.plot(train.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

model.save('/content/drive/MyDrive/AI_Final/ThuMucLuuTru/Data_DuDoanModel_tien.h5')

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from keras.utils import load_img
from keras.utils.image_utils import img_to_array
folder = '/content/drive/MyDrive/AI_Final/ThuMucTest/'
for file in listdir(folder):
  photo = load_img(folder  +  file)
  plt.imshow(photo)
  
  photo = load_img(folder +   file, target_size = (96, 72))
  photo=img_to_array(photo)
  photo=photo.astype('float32')
  photo=photo/255
  photo=np.expand_dims(photo,axis=0)
  result=(model.predict(photo).argmax())
  print(result)
  class_name=['', 'day la con trai 01', 'day la con trai 3', 'day la con trai 10', 'man_11', 'con trai 15', 'day la con trai 20 ', 'day la con trai 22', 'day la con trai 28', 'day la con trai 32', 'day la con trai 36', 'day la con trai 40', 'man_47','man_60','woman_002','day la em gai xinh dep tuoi 4','day la em gai xinh dep tuoi 10','day la em gai xinh dep tuoi 11','day la em gai xinh dep tuoi 14','day la em gai xinh dep tuoi 15','day la em gai xinh dep tuoi 21','day la em gai xinh dep tuoi 22','day la em gai xinh dep tuoi 23','day la em gai xinh dep tuoi 25','day la em gai xinh dep tuoi 32','day la em gai xinh dep tuoi 38']
  print(class_name[result])
  plt.show()