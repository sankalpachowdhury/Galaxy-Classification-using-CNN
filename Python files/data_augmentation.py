from modules import *

#import Augmentor
#train_dir_dest = '/content/Data1/Train/'
#validation_dir_dest = '/content/Data1/Test/'
#train_dir_src = '/content/data/train/'
#validation_dir_src = '/content/data/validation/'


# Data augmentation for Training set
def images_augment(train_dir_src, validation_dir_src, train_dir_dest, validation_dir_dest, galaxy_ids, galaxy_names):
    for serial, galaxy_class in enumerate(galaxy_ids):
        p = Augmentor.Pipeline(source_directory = train_dir_src + galaxy_names[serial],output_directory = train_dir_dest+galaxy_names[serial])
        #Defining methods of Image Augmentation
        p.rotate90(probability=0.5)
        p.rotate270(probability=0.5)
        p.flip_left_right(probability=0.8)
        p.flip_top_bottom(probability=0.3)
        p.resize(probability=1.0, width=150, height=150)

        p.sample(8000)

        # Data augmentation for Validation set
    for serial, galaxy_class in enumerate(galaxy_ids):
        v = Augmentor.Pipeline(source_directory = validation_dir_src + galaxy_names[serial],output_directory = validation_dir_dest+galaxy_names[serial])
        #Defining methods of Image Augmentation
        v.rotate90(probability=0.5)
        v.rotate270(probability=0.5)
        v.flip_left_right(probability=0.8)
        v.flip_top_bottom(probability=0.3)
        v.resize(probability=1.0, width=150, height=150)

        v.sample(1000)
