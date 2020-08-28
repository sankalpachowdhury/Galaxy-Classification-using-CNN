from modules import *
# from keras import models

# img_path = '/content/Data1/Test/elliptical/elliptical_original_107435.jpg_4d3bfbca-e43e-4fbf-bfd4-ec6f73430ac6.jpg' # Path of the image to be passed through the layers
# intermediate layers visualization
def architecture_vis(img_path, model, layer_counts):
   layer_outputs = [layer.output for layer in model.layers[:layer_counts]] # Extracts the outputs of the layers
   activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
   img = image.load_img(img_path)
   img_tensor = image.img_to_array(img)
   img_tensor = np.expand_dims(img_tensor, axis=0)
   activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation
   layer_names = []
   for layer in model.layers[:layer_counts]:
     layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
   images_per_row = 16

   for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
     n_features = layer_activation.shape[-1] # Number of features in the feature map
     size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
     n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
     display_grid = np.zeros((size * n_cols, images_per_row * size))
     for col in range(n_cols): # Tiles each filter into a big horizontal grid
         for row in range(images_per_row):
           channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
           channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
           channel_image /= channel_image.std()
           channel_image *= 64
           channel_image += 128
           channel_image = np.clip(channel_image, 0, 255).astype('uint8')
           display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
     scale = 1. / size
     plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
     plt.title(layer_name)
     plt.grid(False)
     plt.imshow(display_grid, aspect='auto', cmap='viridis')
