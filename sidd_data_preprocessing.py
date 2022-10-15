import os, glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from IPython import display
import mat73
import scipy.io
from skimage import io, transform
import matplotlib.patches as patches


path_sidd_medium_raw = '/dataset/SIDD/SIDD_Medium_Raw/Data'


list_raw_gt    = glob.glob(os.path.join(path_sidd_medium_raw, '*/*GT_RAW*.MAT'), recursive=True)
list_raw_noisy = glob.glob(os.path.join(path_sidd_medium_raw, '*/*NOISY_RAW*.MAT'), recursive=True)
list_metadata  = glob.glob(os.path.join(path_sidd_medium_raw, '*/*METADATA*.MAT'), recursive=True)

list_raw_gt.sort()
list_raw_noisy.sort()
list_metadata.sort()
# for l in list_raw_gt:
#     print(l)
print(len(list_raw_gt))
print(len(list_raw_gt))
print(len(list_metadata))

for g, n, m, in zip(list_raw_gt, list_raw_noisy, list_metadata):
    print('', g, '\n', n, '\n', m)
    break
    
    

m = next(iter(list_metadata))



for k, v in meta.items():
    print(k, v)
    





# Save as numpy
class Rect():
    def __init__(self, x, y, width, height):
        self.x=x
        self.y=y
        self.width=width
        self.height=height

def show_image(image, sz=500):
    fig, ax = plt.subplots(3, 2, figsize=(15,15))
    ax[0][0].imshow(image)
    
    
    colors= ['gray', 'red', 'green', 'blue', 'purple']
    
    r1 =  Rect(1800, 750, sz, sz)
    r2 =  Rect(2800, 1200, sz, sz)
    r3 =  Rect(3400, 1200, sz, sz)
    r4 =  Rect(4000, 1200, sz, sz)
    r5 =  Rect(1800, 2200, sz, sz)
    coors = [r1, r2, r3, r4, r5]
    
    rects = []
    for coor, col in zip(coors, colors):
        r = patches.Rectangle((coor.x, coor.y), coor.width, coor.height, linewidth=1,
                             edgecolor=col, facecolor="none")  
        ax[0][0].add_patch(r)
    
    axs = [ax[0][1], ax[1][0], ax[1][1], ax[2][0], ax[2][1]]
    for ax, coor, col in zip(axs, coors, colors):
        ax.imshow(image[coor.y:coor.y+coor.height,
                        coor.x:coor.x+coor.width,:], filternorm=False, resample=False)

    
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
          
          
          






def load_raw(f):
        raw = mat73.loadmat(f) 
        raw = raw['x']
        return raw
    
def load_metadata(f):
    meta = scipy.io.loadmat(f)
    keys=meta['metadata'][0,0].dtype.names
    vals=meta['metadata'][0,0]
    meta = {k:v for k,v in zip(keys, vals)}
    return meta


def make_raw2grbg(raw, f):
    if '_S6' in f:
        tag = 'S6'
        pass
    elif '_IP' in f:
        tag = 'IP'
        raw    = raw[:, 1:-1]
    elif '_GP' in f:
        tag = 'GP'
        raw    = raw[1:-1, :]
    elif '_N6' in f:
        tag = 'N6'
        raw    = raw[1:-1, :]
    elif '_G4' in f:
        tag = 'G4'
        raw    = raw[1:-1, :]
    else:
        print('unknown class')
        exit()
    return raw


def apply_wb(raw, wb, f):
    import copy
    raw2 = copy.deepcopy(raw) 
    
    raw2[0::2, 1::2] = raw2[0::2, 1::2]*wb[0]
    raw2[1::2, 0::2] = raw2[1::2, 0::2]*wb[2]
        
    return raw2


    
def display(raw, gamma=0.25, title=''):
    img_s_gb = raw[0::2, 0::2]
    img_s_r  = raw[0::2, 1::2]
    img_s_b  = raw[1::2, 0::2]
    img_s_gr = raw[1::2, 1::2]

    image_r  = transform.resize(img_s_r,  (img_s_r.shape[0]*2,  img_s_r.shape[1]*2),  anti_aliasing=True)
    image_b  = transform.resize(img_s_b,  (img_s_b.shape[0]*2,  img_s_b.shape[1]*2),  anti_aliasing=True)
    image_gr = transform.resize(img_s_gr, (img_s_gr.shape[0]*2, img_s_gr.shape[1]*2), anti_aliasing=True)
    image_gb = transform.resize(img_s_gb, (img_s_gb.shape[0]*2, img_s_gb.shape[1]*2), anti_aliasing=True)
    image_g = (image_gr+image_gb)/2

    image_restored = np.concatenate((image_r[..., np.newaxis], 
                                     image_g[..., np.newaxis],
                                     image_b[..., np.newaxis], ), 
                                    axis=2)
    plt.figure(figsize=(16,16))
    plt.imshow(image_restored**gamma)
    plt.title(title)
    plt.show()
#     show_image(image_restored**.25, sz=100)
    
    








def save_as_numpy(fgt, fniosy, fmeta):
    # load data
    raw_gt = load_raw(fgt)    
    raw_noisy = load_raw(fnoisy)
    metadata = load_metadata(fmeta)
    
    # get resolution and reshape
    width  = metadata['Width'].item(0)
    height = metadata['Height'].item(0)
    
    raw_gt    = raw_gt.reshape((height, width))
    raw_noisy = raw_noisy.reshape((height, width))
    
    
#     # get white balance
    wb = 1/metadata['AsShotNeutral'].reshape(-1)
    
    
#     print('raw_gt.shape',raw_gt.shape)
#     print('raw_noisy.shape',raw_noisy.shape)
    
    raw_gt = make_raw2grbg(raw_gt, fgt)
    raw_noisy = make_raw2grbg(raw_noisy, fnoisy)
    
#     display(raw_gt, gamma=0.25, title='before wb')
    
    raw_gt_wb    = apply_wb(raw_gt,    wb, fgt)    
    raw_noisy_wb = apply_wb(raw_noisy, wb, fnoisy)    
    
#     display(raw_gt_wb, gamma=0.25, title='after wb')
     
#     print('raw_gt_wb2.shape',raw_gt_wb.shape)
#     print('raw_noisy_wb2.shape',raw_noisy_wb.shape)
    print(fgt)
    print(fnoisy)
    
    
    def save_it(raw, f, wb):
    
        fname = f.split('/')[-1]
        idx = f.find(fname)
#         print('--> ',f[:idx-1], fname)
        fname = fname[:-4] + f"_Rgain{wb[0]:.3f}_Bgain{wb[2]:.3f}.npy"
        fname = os.path.join(f[:idx-1], fname)
#         print('==>   ',fname)
        np.save(fname, raw)
        
        
#     print('fgt', fgt)       
    save_it(raw_gt,    fgt,    wb)
#     print('fnoisy', fnoisy)   
    save_it(raw_noisy, fnoisy, wb)
    
#     raw_gt_wb.save(fname_gt)
#     np.save(fname_gt, raw_gt_wb)
    
    
    
    fname_gt = fgt.split('/')[-1]
    
     

    return



for idx, (fgt, fnoisy, fmeta) in enumerate(zip(list_raw_gt, list_raw_noisy, list_metadata)):
    print(idx)   
#     if 'S6' in fgt:
#         continue
    save_as_numpy(fgt, fnoisy, fmeta)
#     break










