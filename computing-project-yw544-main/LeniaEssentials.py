#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import re
import psutil
from fractions import Fraction
from scipy.stats import entropy
from skimage.measure import label, regionprops, find_contours
from skimage.transform import rotate
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import center_of_mass
import itertools
from tqdm import tqdm
import copy, re, itertools, json, csv


def figure_world(A, cmap='viridis'):
  fig = plt.figure()
  img = plt.imshow(A, cmap=cmap, interpolation="nearest", vmin=0)
  plt.title('world A')
  plt.close()
  return fig, img

def figure_asset(K, growth, cmap='viridis', K_sum=1, bar_K=False, R=1):
  K_size = K.shape[0];  K_mid = K_size // 2
  fig, ax = plt.subplots(1, 3, figsize=(14,2), gridspec_kw={'width_ratios': [1,1,2]})
  ax[0].imshow(K, cmap=cmap, interpolation="nearest", vmin=0)
  ax[0].title.set_text('kernel K')
  if bar_K:
    ax[1].bar(range(K_size), K[K_mid,:], width=1)
  else:
    ax[1].plot(range(K_size), K[K_mid,:])
  ax[1].title.set_text('K cross-section')
  ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])
  if K_sum <= 1:
    x = np.linspace(0, K_sum, 1000)
    ax[2].plot(x, growth(x))
  else:
    x = np.arange(K_sum + 1)
    ax[2].step(x, growth(x))
  ax[2].axhline(y=0, color='grey', linestyle='dotted')
  ax[2].title.set_text('growth G')
  return fig

def figure_asset_list(Ks, nKs, growth, kernels, use_c0=False, cmap='viridis', K_sum=1):
  global R
  K_size = Ks[0].shape[0];  K_mid = K_size // 2
  fig, ax = plt.subplots(1, 3, figsize=(14,2), gridspec_kw={'width_ratios': [1,2,2]})
  if use_c0:
    K_stack = [ np.clip(np.zeros(Ks[0].shape) + sum(K/3 for k,K in zip(kernels,Ks) if k['c0']==l), 0, 1) for l in range(3) ]
  else:
    K_stack = Ks[:3]
  ax[0].imshow(np.dstack(K_stack), cmap=cmap, interpolation="nearest", vmin=0)
  ax[0].title.set_text('kernels Ks')
  X_stack = [ K[K_mid,:] for K in nKs ]
  ax[1].plot(range(K_size), np.asarray(X_stack).T)
  ax[1].title.set_text('Ks cross-sections')
  ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])
  x = np.linspace(0, K_sum, 1000)
  G_stack = [ growth(x, k['m'], k['s']) * k['h'] for k in kernels ]
  ax[2].plot(x, np.asarray(G_stack).T)
  ax[2].axhline(y=0, color='grey', linestyle='dotted')
  ax[2].title.set_text('growths Gs')
  return fig

def figure_panels(As, Ks, cmap='viridis'):
  global img1, img2, img3, img4
  A_size = As[0].shape[0]
  K_stack = [ np.clip(np.zeros(Ks[0].shape) + sum(K/3 for k,K in zip(kernels,Ks) if k['c0']==l), 0, 1) for l in range(3) ]
  fig = plt.figure(figsize=(8,8), dpi=75, frameon=False)
  plt.subplot(221).set_axis_off();  img1 = plt.imshow(np.dstack(As), vmin=0)
  plt.subplot(222).set_axis_off();  img2 = plt.imshow(np.dstack([np.zeros([A_size, A_size])]*3), vmin=0)
  plt.subplot(223).set_axis_off();  img3 = plt.imshow(np.dstack([np.zeros([A_size, A_size])]*3), vmin=0)
  plt.subplot(224).set_axis_off();  img4 = plt.imshow(np.dstack(K_stack), vmin=0)
  fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
  plt.close()
  return fig

# In[ ]:




