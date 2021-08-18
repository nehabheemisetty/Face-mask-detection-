#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


# In[2]:


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# In[3]:


def compare_images(imageA, imageB, title):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # setup the figure
   # fig = plt.figure(title)
    #plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    #print(title);
    # show first image
   # ax = fig.add_subplot(1, 2, 1)
    #plt.imshow(imageA, cmap = plt.cm.gray)
    #plt.axis("off")
    # show the second image
    #ax = fig.add_subplot(1, 2, 2)
    #plt.imshow(imageB, cmap = plt.cm.gray)
    #plt.axis("off")
    # show the images
   # plt.show()
    return m


# In[4]:


# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("original.JPG")

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# In[5]:


dataset= pd.read_csv('data.csv',sep=',')
data = dataset.iloc[:, :]
data
x = data.iloc[:, :-1].values 
d = dataset.iloc[:, 2]
print(d[2])


# In[6]:


values = [];
for i in range(0,len(d)):
    image = cv2.imread(str(d[i]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values.append(compare_images(original, image, "Original vs."+str(d[i])) )
min=values[0];
for i in range(1,len(d)):
    if min>values[i]:
        min=i 
print(x[min])


fig = plt.figure("Match")
#plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
#plt.suptitle(x[min]) 
plt.suptitle(str(x[min]))
#print(title);
# show first image
ax = fig.add_subplot(1, 2, 1)
plt.imshow(original, cmap = plt.cm.gray)
plt.axis("off")
# show the second image
ax = fig.add_subplot(1, 2, 2)
image = cv2.imread(str(d[min]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image, cmap = plt.cm.gray)
plt.axis("off")
# show the images
plt.show()
#print(dict[min]);