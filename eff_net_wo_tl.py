import tensorflow as tf
from tensorflow.keras import layers
import os
import tensorflow_hub as hub

batch_size = 32
img_height = 200 
img_width = 200 

training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Affwild2/aff_train', 
    validation_split = None, 
    subset = None, 
    image_size = (img_height, img_width), 
    batch_size = 32)
testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Affwild2/aff_val',
    validation_split = None, 
    subset = None, 
    image_size = (img_height, img_width), 
    batch_size = 32)
class_names = training_ds.class_names
print(class_names)
print(len(class_names))
# configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
 
  # while training the model
# prefetch() overlaps data preprocessing and model execution while training
num_classes = len(class_names)
training_ds = training_ds.cache().prefetch(buffer_size = AUTOTUNE) 
testing_ds = testing_ds.cache().prefetch(buffer_size = AUTOTUNE) 


# One-hot encode the labels
def one_hot_encode(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

# Apply one-hot encoding to the dataset
training_ds = training_ds.map(one_hot_encode)
testing_ds = testing_ds.map(one_hot_encode)

#set callback to stop training when accuracy reached 85%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>=0.850):
            print("\nReached 85% accuracy so cancelling training!")
            self.model.stop_training = True
            
            
callbacks = myCallback()

from tensorflow.keras.applications.efficientnet import preprocess_input 
target_size =  (224,224) 

def preprocess_image(image,label): 
    image = tf.image.resize(image, target_size) 
    image = preprocess_input(image) 
    return image, label 

# Assuming you have a dataset named training_ds_resized
training_ds_resized = training_ds.map(preprocess_image) 
testing_ds_resized = testing_ds.map(preprocess_image) 
# Get the shape of a sample image from the dataset
sample_image, _ = next(iter(testing_ds_resized.take(1)))
sample_image_shape = sample_image.shape

print("Shape of a sample image in training_ds_resized:", 
sample_image_shape)

from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

batch_size = 32
img_height = 224
img_width = 224 

inputs = layers.Input(shape=(img_width,img_height,3), batch_size = batch_size) 

#using model without transfer learnaing
outputs = EfficientNetB0(include_top = True, weights = None, classes = num_classes)(inputs) 
eff_wo_tl = tf.keras.Model(inputs,outputs) 
eff_wo_tl.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) 
eff_wo_tl.summary() 
eff_wo_tl.fit(training_ds_resized, validation_data = testing_ds_resized, 
batch_size = batch_size, epochs = 20, callbacks = [callbacks]) 
eff_wo_tl.save('model_weights/effnet_wo_tl.h5')


