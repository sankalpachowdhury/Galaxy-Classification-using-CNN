
# import the data frame
from modules import *

# Decision tree to segregate the data
def dataframe_segregator(df):
    elliptical = df[(df['Class1.1']>0.8) & (df['Class7.1']>0.4)]['GalaxyID'].tolist()
    lenticular = df[(df['Class1.1']>0.8) & (df['Class7.2']>0.4)]['GalaxyID'].tolist()
    spiral = df[(df['Class1.2']>0.8) & (df['Class2.1']>0.4)]['GalaxyID'].tolist()
    galaxy_names = ['elliptical','lenticular','spiral']
    galaxy_ids = [elliptical, lenticular, spiral]

    for serial, galaxy_class in enumerate(galaxy_ids):
        print('Total number of {0} examples: {1}'.format(galaxy_names[serial],len(galaxy_class)))
    return galaxy_ids, galaxy_names

def _proc_images(src, dst, label, arr, percent):
    train_dir = os.path.join(dst, 'train')
    val_dir = os.path.join(dst, 'validation')
    
    train_dest = os.path.join(train_dir, label)
    val_dest   = os.path.join(val_dir, label)
    
    if not os.path.exists(train_dest):
        os.makedirs(train_dest)

    if not os.path.exists(val_dest):
        os.makedirs(val_dest)
    
    random.shuffle(arr)
    
    idx = int(len(arr)*percent)
    for i in arr[0:idx]:
        shutil.copyfile(os.path.join(src, str(i)+'.jpg'), os.path.join(train_dest, str(i)+'.jpg'))
    for i in arr[idx:]:
        shutil.copyfile(os.path.join(src, str(i)+'.jpg'), os.path.join(val_dest, str(i)+'.jpg'))
    
    print(label, 'done!')

# Segregating images for the individual labeled galaxies
#source_path = '/content/images_training_rev1'
#dest_path = '/content/data'

def data_seg(source_path, dest_path, galaxy_ids, galaxy_names):
  for serial, galaxy_class in enumerate(galaxy_ids):
    _proc_images(source_path, dest_path, galaxy_names[serial], galaxy_class, 0.9)
