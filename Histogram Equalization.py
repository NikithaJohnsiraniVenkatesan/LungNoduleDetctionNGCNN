#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as skie
get_ipython().run_line_magic('matplotlib', 'inline')

img = plt.imread(r'C:\Users\nclab\Music\defense preparation\thesis\figures\ch06\34.png', 0)
def show(img):
    # Display the image.
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(12, 3))

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_axis_off()

    # Display the histogram.
    ax2.hist(img.ravel(), lw=0, bins=256)
    ax2.set_xlim(0, img.max())
    ax2.set_yticks([])

    plt.show()


# In[7]:


show(img)


# In[11]:


show(skie.rescale_intensity(img, in_range=(0.4,.95), out_range=(0,1)))


# In[9]:


show(skie.equalize_adapthist(img))


# In[ ]:




