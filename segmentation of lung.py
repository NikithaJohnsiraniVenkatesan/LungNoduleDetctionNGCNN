#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np 
import pandas as pd
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom as dicom
import scipy.misc
import numpy as np
import os
print(os.listdir(r"C:\Users\nclab\Music\defense preparation\dadaset\TCGA-34-2609"))
      
lung = dicom.read_file(r'C:\Users\nclab\Music\defense preparation\dadaset\TCGA-34-5239\1.3.6.1.4.1.14519.5.2.1.6450.4012.229279306490668393002808592693\1.3.6.1.4.1.14519.5.2.1.6450.4012.330436715987009934625711499239\000075.dcm')
slice = lung.pixel_array
slice[slice == -2000] = 0
plt.imshow(slice, cmap=plt.cm.gray)


# In[130]:


def read_ct_scan(folder_name):
        slices = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        
        slices = np.stack([s.pixel_array for s in slices])
        slices[slices == -2000] = 0
        return slices


# In[131]:


ct_scan = read_ct_scan(r"C:\Users\nclab\Music\defense preparation\dadaset\TCGA-34-5239\1.3.6.1.4.1.14519.5.2.1.6450.4012.229279306490668393002808592693\1.3.6.1.4.1.14519.5.2.1.6450.4012.330436715987009934625711499239\\") 


# In[132]:


def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.gray) 


# In[133]:


plot_ct_scan(ct_scan)


# In[134]:


def get_segmented_lungs(im, plot=False):
    

    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))

    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 

    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 

    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 

    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 

    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 

    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 

    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 

    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 
        
    return im


# In[135]:


get_segmented_lungs(ct_scan[20], True)


# In[119]:


def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


# In[120]:


segmented_ct_scan = segment_lung_from_ct_scan(ct_scan)
plot_ct_scan(segmented_ct_scan)


# In[121]:


segmented_ct_scan[segmented_ct_scan < 60] = 0
plot_ct_scan(segmented_ct_scan)


# In[122]:


selem = ball(2)
binary = binary_closing(segmented_ct_scan, selem)

label_scan = label(binary)

areas = [r.area for r in regionprops(label_scan)]
areas.sort()

for r in regionprops(label_scan):
    max_x, max_y, max_z = 0, 0, 0
    min_x, min_y, min_z = 1000, 1000, 1000
    
    for c in r.coords:
        max_z = max(c[0], max_z)
        max_y = max(c[1], max_y)
        max_x = max(c[2], max_x)
        
        min_z = min(c[0], min_z)
        min_y = min(c[1], min_y)
        min_x = min(c[2], min_x)
    if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
        for c in r.coords:
            segmented_ct_scan[c[0], c[1], c[2]] = 0
    else:
        index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))


# In[123]:


def plot_3d(image, threshold=-300):

    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# In[124]:


plot_3d(segmented_ct_scan, 604)


# In[ ]:




