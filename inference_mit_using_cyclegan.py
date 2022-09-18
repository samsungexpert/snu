#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[30]:


import glob, tqdm


# In[19]:


import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import PIL.Image as Image


# In[6]:


model_name = 'cycle_gan'
model_sig  = 'bayer'
base_dir  = '/home/team01/datasets/mit'


## find trining dataset
src_dir = os.path.join(base_dir, 'images', 'train')
folders = glob.glob(os.path.join(src_dir, '**', '**'))
files = []
folders.sort()
for temp in folders:
    folder_num = temp.split('/')[-1]
#     print(temp, folder_num)
    if int(folder_num) % 10 == 1:
        print(temp)
        files += glob.glob(os.path.join(temp, '**/**.png'), recursive=True)

print(files[-1])


# In[7]:


dirs = os.listdir(src_dir)
for idx, d in enumerate(dirs):
    print(idx, d)


# In[9]:


# target folders
new_name = 'images' + '_'  +  model_name + '_' + model_sig
target_dir = os.path.join(base_dir,new_name, 'train')
os.makedirs(target_dir, mode=777, exist_ok=True)


# In[10]:


target_dir


# In[14]:


## model
print(src_dir)
print(target_dir)

name_structure = os.path.join('model_dir', 'checkpoint', model_name + '_G_model_structure.h5')
ckpt_path = os.path.join('model_dir', 'checkpoint',  model_name+'_'+model_sig )
print(name_structure)
print(ckpt_path)
checkpoints = glob.glob(os.path.join(ckpt_path, '*.h5'))
checkpoints.sort()


# In[20]:


model = tf.keras.models.load_model(name_structure, custom_objects={'tf':tf, "InstanceNormalization":tfa.layers.InstanceNormalization})
model.load_weights(checkpoints[-1])
model.summary()
print(checkpoints[-1])


# In[22]:


print(checkpoints[-1])


# In[ ]:


# inference & save
ntot = len(files)
for idx, f in tqdm.tqdm(enumerate(files)):
    name = f.split('train')[-1]
    fsave = os.path.join(target_dir, name[0:])
    fsave = target_dir + name[:-4]
#     print('---->', name)
#     print('====>', fsave)
#     print('=-=->', target_dir)

    image = Image.open(f)
    arr = np.array(image)
#     print(arr.shape, np.amin(arr), np.amax(arr))

    # normalize
    arr = arr.astype(np.float32) / 255.
    arr = arr*2 -1

    # inference
    pred = model.predict(arr[np.newaxis,...]) # (-1, 1) --> (-1, 1)

    # expand (-1, 1) -> (0, 65535)
    pred = pred[0]
    pred = (pred +1) / 2 # (-1, 1) -> (0, 1)
    pred = pred * 65535
    pred = pred.astype(np.uint16)
#     print('>>>>> ',fsave[:-7])
#     print(pred.shape, np.amin(pred), np.amax(pred), pred.dtype)

    os.makedirs(fsave[:-7], mode=777, exist_ok=True)
    np.save(fsave, pred)


#     break


# In[28]:


get_ipython().system('pip install tqdm')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




