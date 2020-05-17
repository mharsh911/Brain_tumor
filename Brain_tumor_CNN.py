#importing the libraries  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#initialising the cnn
classifier = Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('BrainTumor_dataset/train_data',target_size=(128,128),batch_size=16,class_mode='binary')

test_set = test_datagen.flow_from_directory('BrainTumor_dataset/test_data1',target_size=(128, 128),batch_size=16,class_mode='binary')

classifier.fit(train_set,steps_per_epoch=203,epochs=2,validation_data=test_set,validation_steps=50)

results = classifier.evaluate(test_set)
print(results)
classifier.save('CNN.model')