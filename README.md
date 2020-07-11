# Galaxy-Classification-using-CNN

-Sayan Hazra & Sankalpa Chowdhury

Imp links:
1. https://blog.galaxyzoo.org/category/paper/
2. Kaggle Galaxy zoo competition guide: https://github.com/benanne/kaggle-galaxies


#Problem Statement :
**Classify Galaxy**

Data : Image file (.jpeg) of 424 x 424 RGB
Source : 

Step 1: Data Importing:  

        1. Import Data 
        
        2. Labeling part.
        
        3. Train test split

Step 2: Data Pre-processing :

        1. Resize images down to A[0] 69, 69, 3
        
        2. Rotation (Uniform) random [0 ,360] (degree)
        
        3. Translation --> random uniform -4 to +4
        
        4. Zoom --> 1/1.3 to 1.3 Log uniform
        
        5. Flip --> Yes or No, Bernoulli sequence
        
        6. * Colour Peturbation (PCA) and Realtime Augmentation
        
        7. Centering and rescaling --> 
        
Step 3: Model Architecture :

        sequential Keras model
        
        model --> input layer : 69, 69, 3 
        
        layer 1 : Conv3d :  69, 69, 3  filter size: ??
        
        layer 2 : Pooling : 
        
        layer 3 : Conv
        
        layer 4 : Pooling
        
        layer 5 : Conv
        
        layer 6 : Pooling
