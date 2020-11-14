#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
get_ipython().run_line_magic('matplotlib', 'inline')
image = cv2.imread(r"C:\Users\nclab\Music\defense preparation\thesis\figures\ch06\34.png") # reads the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
figure_size = 9 



new_image = cv2.GaussianBlur(image, (figure_size, figure_size),0)
plt.figure(figsize=(11,6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Gaussian Filter')
plt.xticks([]), plt.yticks([])
plt.show()


# In[47]:


import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r"C:\Users\nclab\Music\defense preparation\thesis\figures\ch06\gaussian\56.png")

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()


# In[14]:


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


# In[16]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread(r"C:\Users\nclab\Music\defense preparation\thesis\figures\ch06\34.png",0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.show()


# In[35]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r"C:\Users\nclab\Music\defense preparation\thesis\figures\ch06\34.png",0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


# In[36]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r"C:\Users\nclab\Music\defense preparation\thesis\figures\ch06\34.png", 0)

print('Image Size: ',img.size)

print('Image Shape: ',img.shape)

pix = np.array(img)
print('Image Pixel values: \n',pix)

pix2 = np.copy(pix)

rk, nk = np.unique(pix, return_counts=True)
print('Unique Pixel values: \n',rk)
print('Frequency of pixel values: \n',nk)

pk = nk/img.size
print('pdf: \n',pk)

pk_length = len(pk)
print('Length of pdf: ',pk_length)

sk = np.cumsum(pk)
print('Cumulative Sum: \n',sk)
mul = sk*np.max(pix)
print('Multiplying by Max of Image Pixels: \n',mul)

roundVal = np.round(mul)
print('Rounded value of multiplied value: \n',roundVal)

for i in range(len(pix)):
    for j in range(len(pix[0])):
        pix2[i][j] = roundVal[np.where(rk == pix[i][j])]

print('Old Image: \n', pix)   
print('New Image: \n', pix2)

plt.hist(img.ravel(), 256, [0,256])
plt.show()

plt.hist(pix2.ravel(), 256, [0,256])
plt.show()

plt.imshow(pix, cmap='gray', interpolation='nearest')
plt.show()

plt.imshow(pix2, cmap='gray', interpolation='nearest')
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r"C:\Users\nclab\Music\defense preparation\thesis\figures\ch06\100.png", 0)

print('Image Size: ',img.size)

print('Image Shape: ',img.shape)

pix = np.array(img)
print('Image Pixel values: \n',pix)

pix2 = np.copy(pix)

rk, nk = np.unique(pix, return_counts=True)
print('Unique Pixel values: \n',rk)
print('Frequency of pixel values: \n',nk)

pk = nk/img.size
print('pdf: \n',pk)

pk_length = len(pk)
print('Length of pdf: ',pk_length)

sk = np.cumsum(pk)
print('Cumulative Sum: \n',sk)
mul = sk*np.max(pix)
print('Multiplying by Max of Image Pixels: \n',mul)

roundVal = np.round(mul)
print('Rounded value of multiplied value: \n',roundVal)

for i in range(len(pix)):
    for j in range(len(pix[0])):
        pix2[i][j] = roundVal[np.where(rk == pix[i][j])]

print('Old Image: \n', pix)   
print('New Image: \n', pix2)

plt.hist(img.ravel(), 256, [0,256])
plt.show()

plt.hist(pix2.ravel(), 256, [0,256])
plt.show()

plt.imshow(pix, cmap='gray', interpolation='nearest')
plt.show()

plt.imshow(pix2, cmap='gray', interpolation='nearest')
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:





# In[4]:





# In[ ]:




