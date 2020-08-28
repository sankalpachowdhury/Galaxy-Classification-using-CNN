from modules import *
 
# images visualization
#galaxy_names = ['elliptical','lenticular','spiral']
#folders = ['train', 'validation']
#data_path1 = 'data'
#data_path2 = 'Data1' 

def images_vis(folders, galaxy_names, data_path):
    fig1 = plt.figure(figsize=(15, 15))
    for folder in folders:
        for x,name in enumerate(galaxy_names):
            fig = plt.figure(figsize=(15, 15))
            for i,j in enumerate(os.listdir('/{0}/{1}/{2}/'.format(data_path, folder, name))[0:3]):
                fig.add_subplot(3, 3,i+1)    
                plt.imshow(img.imread(os.path.join('/{0}/{1}/{2}/'.format(data_path, folder, name),j)))
                fig.suptitle('\nSamples of {0} images correponding to galaxy category {1}'.format(folder, name),fontsize = 14)
    plt.show()
    
