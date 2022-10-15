import numpy as np 
import os,glob



folder_orig = '/data/team19/pixelshift/PixelShift200_train/'
filepath = glob.glob(folder_orig + "*.npy") 

folder_target = '/data/team19/pixelshift/PixelShift200_train_patch/'
os.mkdir(folder_target)


aa = np.load(filepath[0])

H,W, C = aa.shape

p_size = 128


for filename in filepath:
    img_temp = np.load(filename)
    img_name = filename.split("/")[-1].split('.')[0]

    cnt = 0
    for y in range(0, H-p_size, p_size):
        for x in range(0, W-p_size,p_size):
            save_name = folder_target+img_name + '_{0:03d}'.format(cnt)+".npy"
            cnt+=1
            
            img_patch = img_temp[y:y+p_size,x:x+p_size,:]

            with open(save_name,'wb') as f:
                np.save(f, img_patch)

            

           



