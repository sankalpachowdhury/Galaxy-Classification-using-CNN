from modules import *

#import tensorflow as tf
# target = (150, 150) # target image size after Augmentation
# batch_size = 32     # batch size (hyperparameter)
# c_mode = 'categorical'

# creating image_data_generator
def image_generator_create(train_dir, validation_dir, target, batch_size, c_mode):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1.0/255,
                    rotation_range=25,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2)
    validation_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1.0/255,
                    #rotation_range=25,
                    #width_shift_range=.15,
                    #height_shift_range=.15,
                    #horizontal_flip=True,
                    #zoom_range=0.2
                    )
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=target,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode = c_mode)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                         target_size=target,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         class_mode = c_mode)

    return train_generator, validation_generator
