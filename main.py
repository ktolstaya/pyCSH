# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:59:42 2018

@author: Ekaterina Tolstaya

This program is based on the code of "Coherency Sensitive Hashing", Matlab and C++
by Simon Korman and Shai Avidan

http://www.eng.tau.ac.il/~simonk/CSH/

Coherency Sensitive Hashing (CSH) extends Locality Sensitivity Hashing (LSH) 
and PatchMatch to quickly find matching patches between two images. LSH relies 
on hashing, which maps similar patches to the same bin, in order to find matching 
patches. PatchMatch, on the other hand, relies on the observation that images are 
coherent, to propagate good matches to their neighbors, in the image plane. It 
uses random patch assignment to seed the initial matching. CSH relies on hashing 
to seed the initial patch matching and on image coherence to propagate good matches. 
In addition, hashing lets it propagate information between patches with similar 
appearance (i.e., map to the same bin). This way, information is propagated much 
faster because it can use similarity in appearance space or neighborhood in the 
image plane. As a result, CSH is at least three to four times faster than PatchMatch 
and more accurate, especially in textured regions, where reconstruction artifacts 
are most noticeable to the human eye.

--------------------------------------------

The CSH algorithm was applied in my depth propagation project:

https://www.researchgate.net/publication/282681757_Depth_propagation_for_semi-automatic_2D_to_3D_conversion

"""
import numpy as np
import cv2
from utils import getCSHmap

PATCH_WIDTH = 8

def reconstruct_image(f0, CSHnnMapX, CSHnnMapY, sigma):
    f0 = f0/255
    m = f0.shape[0]
    n = f0.shape[1]
    R = np.zeros(f0.shape)
    Rcount = np.zeros((m,n))
    for i in range(m):
       for j in range(n): 
          if 1<=i and i+PATCH_WIDTH-1<=m and 1<=j and j+PATCH_WIDTH-1<=n :
             patch = f0[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1,:]
             i2 = int(CSHnnMapY[i,j])
             j2 = int(CSHnnMapX[i,j])
             patch2 = f0[i2:i2+PATCH_WIDTH-1,j2:j2+PATCH_WIDTH-1,:]
             d = sum( (patch.flatten()-patch2.flatten())**2 )
             coeff = np.exp( -d / (2*sigma**2) )
             R[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1,:] = R[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1,:] + coeff*patch2
             Rcount[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1] = Rcount[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1] + coeff
    
    Rcount = np.tile(Rcount[:,:,np.newaxis],(1,1,3))
    R[Rcount>0] = ( R[Rcount>0]/ Rcount[Rcount>0] )*255

    return R

def generateData(F0,F1,F2, d0,d2, sigma):
    
    startscale = -3
    n_iter = 3
    d1 = d0*0

    for logscale in range(startscale,1):
        scale = 2**logscale
        f0 = cv2.resize(F0,None, fx=scale, fy=scale)
        f1 = cv2.resize(F1,None, fx=scale, fy=scale)
        f2 = cv2.resize(F2,None, fx=scale, fy=scale)
                
        d00 = cv2.resize(d0,(f0.shape[1], f0.shape[0]))
        d01 = cv2.resize(d1,(f0.shape[1], f0.shape[0]))
        d02 = cv2.resize(d2,(f0.shape[1], f0.shape[0]))
    
        for p in range(n_iter):
            
            if not(logscale == startscale):
                if p == 0:
                    f1[:,:,2] = d01
                    f0[:,:,2] = cv2.resize(cv2.resize(d00,(0,0),fx=0.5, fy=0.5),(d00.shape[1],d00.shape[0]))
                    f2[:,:,2] = cv2.resize(cv2.resize(d02,(0,0),fx=0.5, fy=0.5),(d00.shape[1],d00.shape[0]))
                else:
                    f0[:,:,2] = d00
                    f1[:,:,2] = d01
                    f2[:,:,2] = d02
                   
            CSHnnMapX0, CSHnnMapY0 = getCSHmap(f1, f0, PATCH_WIDTH)
            CSHnnMapX2, CSHnnMapY2 = getCSHmap(f1, f2, PATCH_WIDTH)
            
            m = f0.shape[0]
            n = f0.shape[1]
            R = np.zeros(d00.shape)
            Rcount = np.zeros((m,n))
            for i in range(m):
               for j in range(n): 
                  if 1<=i and i+PATCH_WIDTH-1<=m and 1<=j and j+PATCH_WIDTH-1<=n :
                     patch0 = f0[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1,:]/255
                     i0 = int(CSHnnMapY0[i,j])
                     j0 = int(CSHnnMapX0[i,j])
                     patch_a = f1[i0:i0+PATCH_WIDTH-1,j0:j0+PATCH_WIDTH-1,:]/255
                     e0 = sum( (patch0.flatten()-patch_a.flatten())**2 )
                     
                     patch2 = f2[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1,:]/255
                     i2 = int(CSHnnMapY2[i,j])
                     j2 = int(CSHnnMapX2[i,j])
                     patch_b = f1[i2:i2+PATCH_WIDTH-1,j2:j2+PATCH_WIDTH-1,:]/255
                     e2 = sum( (patch2.flatten()-patch_b.flatten())**2 )

                     patch1da = d00[i0:i0+PATCH_WIDTH-1,j0:j0+PATCH_WIDTH-1]/255
                     coeffa = np.exp( -e0 / (2*sigma**2) )
                     patch1db = d02[i2:i2+PATCH_WIDTH-1,j2:j2+PATCH_WIDTH-1]/255
                     coeffb = np.exp( -e2 / (2*sigma**2) )
                    
                     R[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1] = R[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1] + coeffa*patch1da + + coeffb*patch1db
                     Rcount[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1] = Rcount[i:i+PATCH_WIDTH-1,j:j+PATCH_WIDTH-1] + coeffa+coeffb
            
            R[Rcount>0] = ( R[Rcount>0]/ Rcount[Rcount>0] )*255
            d01 = R.copy()
    
        d1 = d01.copy()
        
    return d1  




# # Image synthesis example
# f = cv2.imread('examples/Saba1.bmp')
# f0 = cv2.imread('examples/Saba2.bmp')
# CSHnnMapX, CSHnnMapY = getCSHmap(f, f0, PATCH_WIDTH)
# im = reconstruct_image(f0, CSHnnMapX, CSHnnMapY, 1.5)
# cv2.imwrite('results/SabaResult.png', im)

# Depth propagation example
F0 = cv2.imread('examples/157_LionKing_s01_00000.bmp')
F1 = cv2.imread('examples/157_LionKing_s01_00004.bmp')
F2 = cv2.imread('examples/157_LionKing_s01_00008.bmp')
d0 = cv2.imread('examples/depth_00000.bmp')
d0 = d0[:,:,0]
d2 = cv2.imread('examples/depth_00008.bmp')
d2 = d2[:,:,0]

sigma = 1.5 
   
d = generateData(F0,F1,F2,d0,d2,sigma)
cv2.imwrite('results/depth_00004.png', d)



