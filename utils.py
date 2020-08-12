# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:59:42 2018

@author: Ekaterina Tolstaya

This program is based on the code of "Coherency Sensitive Hashing", Matlab and C++
by Simon Korman and Shai Avidan

http://www.eng.tau.ac.il/~simonk/CSH/
    
"""

import numpy as np
import bisect

FILT_MATR_NUM = 23
HIST_BINS_SHIFT = 10
HIST_BINS = (1 << HIST_BINS_SHIFT)
MIN_BINS = 2 # the minimal number of bins
maxKernels = 3*5*5
numHashs = 2 # width of hash table
iters = 8
maxBits = 18
tableSize = (1<<maxBits)

nSequencyOrder16u = np.array([0,3,9,1,2,12,16,6,17,20,15,13,14,4,5,10,11,7,8,21,22,18,19])

# get most significant bit 
def get_msb(x): 
    r = 0
    if (x < 1):
        return 0
    x >>= 1
    while (x): 
        r+=1
        x >>= 1
    return r

def GetGCKTraverseParams():
    deltas1D = np.array([4,2,4,1,4,2,4])
    alphaDirs1D = np.array([-1,-1,1,-1,-1,1,1])
    snakeOrder = np.array([[1, 2, 2, 1, 1, 2, 3, 3, 3, 4, 4,  4, 4, 3, 2, 1, 1, 2, 
			3,     4,     5,     5,     5,     5 ,    5 ,    6,     6 ,    6 ,    6 ,    6 ,    6 ,    5 ,    4 ,    3 ,    2 ,    1,
			1,     2,     3 ,    4 ,    5 ,    6,     7 ,    7 ,    7 ,    7 ,    7 ,    7 ,    7  ,   8 ,    8 ,    8 ,    8 ,    8,
			8,     8,     8,     7 ,    6 ,    5 ,    4 ,    3 ,    2 ,    1],
            [1,     1 ,    2,     2,     3 ,    3 ,    3,     2,     1,     1,     2,     3,     4,     4,     4,     4,     5,     5, 
			5,     5,     5  ,   4 ,    3,     2  ,   1 ,    1,     2,     3 ,    4 ,    5 ,    6,     6 ,    6 ,    6  ,   6 ,    6,
			7 ,    7 ,    7,     7 ,    7,     7 ,    7 ,    6 ,    5,     4,     3 ,    2,     1,     1,     2 ,    3,     4 ,    5,
			6 ,    7,     8,     8 ,    8 ,    8 ,    8,     8,     8,     8]])  
    return deltas1D, alphaDirs1D, snakeOrder

#/* This function computes a DC rectangle kernel of source size for the source image 
#The function does not assume zero-padded edges exist, but for reasonable results these should be considered
#The result image is the rectangle DC kernel of the source image, anchored on the top-left pixel of the kernel
#The size of the resulting matrix is the same as the source image
#Source image type must be uint8 and must be single channel
#Output image type is int16
#*/
def ApplyPlusKernel8u16s(M, width):
    KernelSize = width; 
    m = M.shape[1]
    n = M.shape[0]
    pIntegralImage = np.zeros((n,m),dtype=np.uint32)
    pIntegralImage[0,0] = M[0,0]
    for i in range(1,m):
         pIntegralImage[0,i] = pIntegralImage[0,i-1] + M[0,i]
    #pIntegralImage[0,i] = pIntegralImage[0,i-1] + M[0,i]
         
    for j in range(1,n):
         nCumSumCurIter = 0
         for i in range(m):
             nCumSumCurIter = nCumSumCurIter  + M[j,i]
             pIntegralImage[j,i] = pIntegralImage[j-1,i] + nCumSumCurIter
	  
    M[0,0] = pIntegralImage[KernelSize-1,KernelSize-1]

    for i in range(KernelSize,m):
          M[0,i-KernelSize+1] = pIntegralImage[KernelSize-1,i] - pIntegralImage[KernelSize-1,i-KernelSize]
     
    for j in range(KernelSize,n):
    		M[j-KernelSize+1,0] = pIntegralImage[j,KernelSize-1] - pIntegralImage[j-KernelSize,KernelSize-1]

    for j in range(KernelSize, n):
          for i in range(KernelSize, m):
             M[j-KernelSize+1,i-KernelSize+1] = pIntegralImage[j,i] - pIntegralImage[j-KernelSize,i] - pIntegralImage[j,i-KernelSize] + pIntegralImage[j-KernelSize,i-KernelSize]

    return M

#/* This function computes the next step in a series of GCK kernels
#The function does not assume zero-padded edges exist, but erroneous results will be produced if these are not considered
#The size of the resulting matrix is the same as the source image
#Source image type must be int16 and must have three channel
#bCurAlphaPos - Bollean defining if current kernel is positive alpha-related relative to last kernel
#nDelta - delta between current and last kernels
#bHorizMovement - Bollean defining if current step's delta is horizontal or vertical 
#Output image type is int16 and single channel; Number of columns is three times the number of columns in a single channel of the source image
#*/
def GCKFilterSingleStep16s_C3(nChannelsToProcess, #/* usually=3 */, 
                               pInputImage16s, source_from, target_from, pKernelSize, 
                               bCurAlphaPos, pDeltaStep, bHorizMovement):
# Get sizes of variables 
   m = pInputImage16s[source_from].shape[1]
   n = pInputImage16s[source_from].shape[0]
	
# Initialize filter parameters
   if (bHorizMovement):
       nFirstRow = 1
       nLastRow  = m
       nFirstCol = pKernelSize
       nLastCol  = n-pKernelSize
       nRowDeltaStep = pDeltaStep
       nColDelta     = 0
   else:
       nFirstRow = pKernelSize
       nLastRow  = m-pKernelSize
       nFirstCol = 1
       nLastCol  = n
       nRowDeltaStep = 0
       nColDelta     = pDeltaStep

   if (bCurAlphaPos):
       for j in range(nFirstCol,nLastCol):
          for i in range(nFirstRow, nLastRow): 
              for ch in range(nChannelsToProcess):      
                   pInputImage16s[target_from + ch][j,i] = pInputImage16s[source_from + ch][j,i] + \
                                                 pInputImage16s[source_from + ch][j - nRowDeltaStep,i-nColDelta] + \
																pInputImage16s[target_from + ch][j - nRowDeltaStep,i-nColDelta]

   else:
       for j in range(nFirstCol,nLastCol):
          for i in range(nFirstRow,nLastRow): 
              for ch in range(nChannelsToProcess):
                 pInputImage16s[target_from + ch][j,i] = pInputImage16s[source_from + ch][j,i] - \
																pInputImage16s[source_from + ch][j - nRowDeltaStep,i-nColDelta] - \
																pInputImage16s[target_from + ch][j - nRowDeltaStep,i-nColDelta]
   return  pInputImage16s

        
#/* This function computes the next step in a series of GCK kernels
#The function does not assume zero-padded edges exist, but erroneous results will be produced if these are not considered
#The size of the resulting matrix is the same as the source image
#Source image type must be int16 and must have three channel
#bCurAlphaPos - Bollean defining if current kernel is positive alpha-related relative to last kernel
#nDelta - delta between current and last kernels
#bHorizMovement - Bollean defining if current step's delta is horizontal or vertical 
#Output image type is int16 and single channel; Number of columns is three times the number of columns in a single channel of the source image
#*/
def ProceedOneSnakeStepFor3ChannelsMexTailored(pInputImage16s, source_from, target_from, 
								snakeOrder, deltas1D, alphaDirs1D, snakeInd, procs, width, yOnly):
   prevInd = procs[snakeInd-1]
   prevY = snakeOrder[1-1,prevInd-1]
   prevX = snakeOrder[2-1,prevInd-1]
   Y = snakeOrder[1-1,snakeInd-1] 
   X = snakeOrder[2-1,snakeInd-1]

   if (Y == prevY):# -> moving horizontally
       if (prevX < X):
          delta_j = deltas1D[prevX-1]; 
          dir_j = alphaDirs1D[prevX-1];		  
       else:
          delta_j = deltas1D[X-1];
          dir_j = -1*alphaDirs1D[X-1];		  
       LRnotUD = True;
	
   else:# (X == prevX) -> moving vertically
       if (prevY < Y):		
             delta_j = deltas1D[prevY-1];
             dir_j = alphaDirs1D[prevY-1];		
       else:
             delta_j = deltas1D[Y-1];
             dir_j = -1*alphaDirs1D[Y-1];
       LRnotUD = False;
	
   bCurAlphaPos = dir_j==1
   if (yOnly):
      pInputImage16s = GCKFilterSingleStep16s_C3(1, pInputImage16s, source_from, target_from, width, bCurAlphaPos, delta_j, LRnotUD)
   else:
      pInputImage16s = GCKFilterSingleStep16s_C3(3, pInputImage16s, source_from, target_from, width, bCurAlphaPos, delta_j, LRnotUD)

   return pInputImage16s

def GetResultsOfKernelApplication(frame, maxKernels, PATCH_WIDTH):

   TLBoundary = 2*PATCH_WIDTH
   deltas1D, alphaDirs1D, snakeOrder = GetGCKTraverseParams()
   filtersY = np.array([1,     4,     7,    10,    13,    16,    19 ,   22  ,  25  ,  28 ,   31  ,  43 ,   46 ,   49 ,   73])
   # these two length 25 = 5x5 vectors will guide the computation during the snake-order scan. They
   # will tell the computation which cell to use as the previous in the computation.
   # the Y-FILTER INDEX from which we come FOR THE COMPUTATION
   procFilterIndToUse = np.array([0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, -1,  -1,  -1,  14,  20,  21, -1, -1,  -1, -1, -1,  -1, -1,  18])
   # the SNAKE INDEX from which we come FOR THE COMPUTATION
   procSnakeIndToUse = np.array([0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10, -1,  -1,  -1,   6,  15,  16, -1, -1,  -1, -1, -1,  -1, -1,  10])
   LastCbCrFilterIndex = 12

   h = frame.shape[0]
   w = frame.shape[1]

   pTmpA = []
   # zeros-padding
   # initialization of first 3 matrixes for filtering
   for mi in range(3):
      M = np.zeros((h + TLBoundary, w + TLBoundary), dtype=np.int16)
      M[TLBoundary:,TLBoundary:] = frame[:,:,mi]
      pTmpA.append(M)

   for t in range(4,FILT_MATR_NUM+1):
      pTmpA.append(np.zeros((h + TLBoundary, w + TLBoundary), dtype=np.int16))
        
   # pFiltA, pFiltB:
   # 0000000000000000000000000000000000
   # 0000000000000000000000000000000000
   # 0000000000000000000000000000000000
   # 00000000xxxxxxxxxxxxxxxxxxxxxxxxxx
   # 00000000xxxxxxxxxxxxxxxxxxxxxxxxxx
   # 00000000xxxxxxxxxxxxxxxxxxxxxxxxxx
   # 00000000xxxxxxxxxxxxxxxxxxxxxxxxxx
   # 00000000xxxxxxxxxxxxxxxxxxxxxxxxxx
   # 00000000xxxxxxxxxxxxxxxxxxxxxxxxxx

   pTmpA[0] = ApplyPlusKernel8u16s(pTmpA[0], PATCH_WIDTH)
   pTmpA[1] = ApplyPlusKernel8u16s(pTmpA[1], PATCH_WIDTH)
   pTmpA[2] = ApplyPlusKernel8u16s(pTmpA[2], PATCH_WIDTH)

   currFilter = 3
   channelInd = 0
   for iteration in range(1,maxKernels+1):
      channelInd = channelInd + 1   # start with r and change cyclicly
      if ((channelInd > 3) and (iteration < maxKernels)):
         channelInd = 1
         cont = False
         for fy in range(0,15):
             cont = iteration == filtersY[fy]
             if (cont):
                break
         if cont:
            yOnly = (iteration > LastCbCrFilterIndex)
            addition = 2*(1-int(yOnly))
            snakeInd = 1 + int(iteration/3) # the index of the filter that we are going source_to
            source_from = procFilterIndToUse[snakeInd-1] - 1
            pTmpA = ProceedOneSnakeStepFor3ChannelsMexTailored(pTmpA, source_from, currFilter, snakeOrder,deltas1D,alphaDirs1D,snakeInd,procSnakeIndToUse,PATCH_WIDTH,yOnly);
            currFilter = currFilter + 1 + addition;        

   # copy inner part of zero-padded matrix (TmpA) into FiltA without zeros.
   pFiltA = []
   for mi in range(FILT_MATR_NUM):
      flt = np.zeros((h,w), dtype=np.int16) 
      for ii in range(h):
         for jj in range(w):
            flt[ii,jj] = pTmpA[mi][ii+ TLBoundary,jj+ TLBoundary]
      pFiltA.append(flt)

   return pFiltA

def ModifyBoundaries(pFiltA,pFiltB, largeVal, width):

   LargeValDiv2 = largeVal>>1
   rows = pFiltB[0].shape[0]
   cols = pFiltB[0].shape[1]

   #  Loop on first size kernels - no image can return these results for projections -
   for p in range(FILT_MATR_NUM): 
      if (p > 0):
         largeVal = LargeValDiv2
      for r in range(rows):
         pFiltB[p][r,0] = largeVal 
         for c in range(cols-width,cols):
            pFiltB[p][r,c] = largeVal
      for r in range(rows - width,rows):
         for c in range(cols):
            pFiltB[p][r,c] =  largeVal
   return pFiltA, pFiltB

def BuildWalshHadamardCodes(width, pFiltA, pFiltB, numTables, ColorChannels, 
								code_A, code_B,    # these are matrices of sizes of A,B holding the calculated code per patch
								bitCountPerTable): # denotes how many bits were used to build this code
   maxFilterRange = 2*256*width*width+1   # always odd
   minFilterValue = -(maxFilterRange/2)
   YCbCrRatio = 8 # the ratio between number of bins we give to Y vs chroma, for a given WH filter

   # 'hand-crafting' the number of bins per filter
   # This is done according to distance of the WH filter from the origin (empirical)
   # index is the snake order (these "number of bins" are designed for y and we use divide by YCbCrRatio for the color channels

   nrepsTabs = 4*width-1
   repsTable = np.zeros(nrepsTabs,dtype=int)
   for t in range(nrepsTabs):
      repsTable[t] = 2

   repsTable[0] = 32
   repsTable[1] = 8
   repsTable[2] = 2
   repsTable[3] = 8
   # 'hand-crafting' the choice of filters
   # these are the WH indices of the 8 kernels we want to work with
   # and these are their LOCATIONS in the currFiltImgs_A structure:
   filters = np.array([1,2,3,4,7,10,13,17,22,23])  

   maxIterations = np.max(filters)
   channelInd = 0
   nextBit = 1;
   numBinsY = repsTable[0]

   kHIST_STEP_A = 21
   kHIST_STEP_B = 13

   # A] BUILDING OF CODES for A and B (iterating through the kernels)
   # at each iteration w`e append to the code the bin index 
   # of the patch projected on the current WH code
	        
   for iteration in range(maxIterations+1):
      channelInd = channelInd + 1 # start with r and change cyclicly
      if (channelInd > 3):
         channelInd = 1
         # need this to know how many bins to use
         snakeInd = int(1 + np.floor(iteration/3)) # the index of the filter that we are going to
         numBinsY = repsTable[snakeInd - 1]
		
	    
      # keep going if this current filter is in the 'filters' list
      cont = False
      for fy in range(8):
         cont = iteration == (filters[fy]-1) 
         if (cont):
            break
      if (cont):				    
         if ( not(iteration%3) ): # current channel is Y channel
            numBins = int(numBinsY)
         else:
            numBins = np.max([MIN_BINS, int(numBinsY/YCbCrRatio + 0.5) ])
				    
       # 1) Build bin map with random offset - this is a lookup-table for binning (bin edges).
       # --------------------------------
       # This is done by taking a relatively small joint sample of patches from A and B,
       # and placing bin edges such that the sample distributes evenly into the bins. In
       # addition, bins are shifted at a random amount (first and last bin won't contain the same amounts)
       # 'binMapForUse' is the lookup table that holds the bin index for each projection value
       # --------------------------------
         maxF = 0
         minF = 32766
         h = pFiltA[iteration].shape[0]
         w = pFiltA[iteration].shape[1]
         for jj in range(h): 
            locMaxA = maxF
            locMinA = minF
            for ii in range(w): 
               if pFiltA[iteration][jj,ii] < maxFilterRange/2:
                  locMaxA = np.max( [pFiltA[iteration][jj,ii], locMaxA] )
                  locMinA = np.min( [pFiltA[iteration][jj,ii], locMinA] )
               if pFiltB[iteration][jj,ii] < maxFilterRange/2:
                  locMaxA = np.max( [pFiltB[iteration][jj,ii], locMaxA] )
                  locMinA = np.min( [pFiltB[iteration][jj,ii], locMinA] )
				
            if (locMaxA > maxF):  maxF = locMaxA
            if (locMinA < minF):  minF = locMinA

         hist_shift = get_msb(int(maxF - minF + 1))
         if (hist_shift >= HIST_BINS_SHIFT): 
            hist_shift -= (HIST_BINS_SHIFT - 1)
         else:
            hist_shift = 0

         hist = np.zeros(HIST_BINS, dtype=np.uint32)
         ind = 0
         h = pFiltA[iteration].shape[0]
         w = pFiltA[iteration].shape[1]
         total = 0
         for ind in range(0,w*h, kHIST_STEP_A):
            if (pFiltA[iteration].ravel()[ind] < maxF):
               val = ( int(pFiltA[iteration].ravel()[ind] - minF) >> hist_shift)
               hist[val] += 1
               total += 1
         for ind in range(0,w*h,kHIST_STEP_B):
            if (pFiltB[iteration].ravel()[ind] < maxF):
               val = (int(pFiltB[iteration].ravel()[ind] - minF) >> hist_shift)
               hist[val] += 1
               total += 1

         edges = np.zeros(numBins+1, dtype=int)
         fact = 100.0/numBins
         shift = fact*(np.random.rand() - 0.5) 
         count = 0
         percentiles = shift
         sum = 0
         h1 = 0
         while ((percentiles < 100) and (count<numBins+1)):
            numEl = max(percentiles, 0) * total / 100.0
            while ((h1 < 1000) and (sum  < numEl)):
               sum += hist[h1]
               h1 += 1
            edges[count] = (h1 << hist_shift) + minF - minFilterValue
            count += 1
            percentiles += fact
         edges[0] = 1
         edges[numBins] = maxFilterRange


         # use binary search for 32 (and more) bins
         if (numBins >= 32):
            for jj in range(h):
               for ii in range(w):
                  d1 = pFiltA[iteration][jj,ii ] - minFilterValue - 1
                  d2 = pFiltB[iteration][jj,ii ] - minFilterValue - 1
                  p1 = bisect.bisect_left(edges, d1, 1, len(edges)-1)
                  bins_A = (int(p1)-1)<<(nextBit-1)
                  p2 = bisect.bisect_left(edges, d2, 1, len(edges)-1)
                  bins_B = (int(p2)-1)<<(nextBit-1)

                  # appending the current bin numbers the forming code
                  code_A[jj,ii] |= bins_A
                  code_B[jj,ii] |= bins_B 
         elif (numBins > 2): 
            for jj in range(h):
               for ii in range(w):
                  d1 = pFiltA[iteration][jj ,ii ] - minFilterValue - 1
                  d2 = pFiltB[iteration][jj ,ii ] - minFilterValue - 1
                  d1 = np.min([d1,maxFilterRange])
                  d2 = np.min([d2,maxFilterRange])

                  inda = 1
                  while (edges[inda] < d1):  
                     inda += 1
                  bins_A = int(inda-1)<<(nextBit-1)
                  indb = 1
                  while (edges[indb] < d2):  
                     indb += 1

                  bins_B = int(indb-1)<<(nextBit-1)  

                  # appending the current bin numbers the forming code
                  code_A[jj,ii] |= bins_A
                  code_B[jj,ii] |= bins_B
         else:
            for jj in range(h):
               for ii in range(w):
                  d1 = pFiltA[iteration][jj ,ii ] - minFilterValue - 1
                  d2 = pFiltB[iteration][jj ,ii ] - minFilterValue - 1
                  
                  if (edges[1] < d1): 
                      inda = 2 
                  else: 
                      inda = 1
                  bins_A = int(inda-1)<<(nextBit-1)

                  if (edges[1] < d2):
                      indb = 2
                  else: 
                      indb = 1
                  bins_B = int(indb-1)<<(nextBit-1);  

                  # appending the current bin numbers the forming code
                  code_A[jj,ii] |= bins_A
                  code_B[jj,ii] |= bins_B

         numBitsForBinning = get_msb(numBins)
         nextBit = nextBit + numBitsForBinning
         if (nextBit > 32):
            break
      bitCountPerTable = nextBit - 1
   return code_A, code_B, bitCountPerTable
	
def hash_1(a):
	a = (a ^ 61) ^ (a >> 16)
	a = a + (a << 3)
	a = a ^ (a >> 4)
	a = a * 0x27d4eb2d
	a = a ^ (a >> 15)
	return a

def hashFun(code_A, maxBits):
    bitMask = (1 << maxBits) - 1
    return hash_1(code_A) & bitMask

def HashCodesIntoTables(hA, wA, maxBits, code_A, hTables_A, indices_A_offset, 
                         hashBits, tableSize, bitCountPerTable):
   NeedToHash = bool(bitCountPerTable >  maxBits)
   if (NeedToHash): 
      hashFun( code_A, maxBits )
	
   hashPtrs = np.zeros(tableSize, dtype=np.uint32)
   for i in range(wA, hA*wA - wA):
      ind_A_flat = code_A.ravel()[i]
      entry = ind_A_flat;
      hTables_A[hashPtrs[entry]][entry] = i
      hashPtrs[entry] = 1
   return hTables_A

def SqrDiff(a,b):
    return (a-b)**2

def CalcSqrDifferenceAndTakeBest(  pCurrBestError32u, pCurrBestMapping32u, nCurrCandMapping32u, nCurIterOffset32u, pFiltA, pFiltB):

   if (pCurrBestMapping32u == nCurrCandMapping32u):
      return pCurrBestError32u, pCurrBestMapping32u
   ReminderBestError = pCurrBestError32u
   sequence_0 = nSequencyOrder16u[0]
   nCurError32u = SqrDiff(pFiltA[sequence_0].ravel()[nCurIterOffset32u], pFiltB[sequence_0].ravel()[nCurrCandMapping32u])

   if (nCurError32u > ReminderBestError):
      return pCurrBestError32u, pCurrBestMapping32u

	# all the rest (in pairs)
   for k in range(1, FILT_MATR_NUM-1, 2):
      sequence_k1 = nSequencyOrder16u[k]
      sequence_k2 = nSequencyOrder16u[k+1]
		
      nCurError32u += SqrDiff(pFiltA[sequence_k1].ravel()[nCurIterOffset32u], pFiltB[sequence_k1].ravel()[nCurrCandMapping32u])
      nCurError32u += SqrDiff(pFiltA[sequence_k2].ravel()[nCurIterOffset32u], pFiltB[sequence_k2].ravel()[nCurrCandMapping32u])
      if (nCurError32u > ReminderBestError):
         return pCurrBestError32u, pCurrBestMapping32u

   if ((FILT_MATR_NUM & 1) == 0): # Even number of kernels, check lust o
      sequence_last = nSequencyOrder16u[FILT_MATR_NUM-1]
      nCurError32u += SqrDiff(pFiltA[sequence_last].ravel()[nCurIterOffset32u], pFiltB[sequence_last].ravel()[nCurrCandMapping32u])
      if (nCurError32u > ReminderBestError):
         return pCurrBestError32u, pCurrBestMapping32u

   pCurrBestError32u   = nCurError32u
   pCurrBestMapping32u = nCurrCandMapping32u 

   return pCurrBestError32u, pCurrBestMapping32u    

# This function performs propagation by neighbours of NN mapping
def GCKCodebookPropagation16s_FilterVectors( bestErrorsNewA, bestMappingNewA, bDownwardFlag, nHashsNoe16u, 
                                            pFiltA, pFiltB, pCodeA, pCodeB, hTables_A, hTables_B):
   rows = pCodeB.shape[0]
   cols = pCodeB.shape[1]
   codeLen = rows*cols
   # Direction of propagation depending on current loop running direction (downwards/upwards)       
   if (bDownwardFlag):
      # Filter downwards
      nFirstRowForward = 1
      nLastRowForward = rows
      nFirstColForward = 1
      nLastColForward = cols
      nLeftRightOffset = 1
      nTopBotOffset = cols
      nLeftRightTargetOffset = 1  
   else:
      nFirstRowForward = 0
      nLastRowForward = rows - 2
      nFirstColForward = 0
      nLastColForward = cols - 2
      nLeftRightOffset = -1
      nTopBotOffset = -cols
      nLeftRightTargetOffset = -1
          
   # Begin looping on input image pixels forwards
   for j in range(nFirstRowForward, nLastRowForward):
      nCurIterStep = j*cols
      for i in range(nFirstColForward, nLastColForward): 
         nCurIterOffset = nCurIterStep+i
         CurrBestError32u   = np.uint32(bestErrorsNewA[j,i])
         CurrBestMapping32u = np.uint32(bestMappingNewA[j,i])

         nHorizNeighbourMapping32u    = bestMappingNewA.ravel()[nCurIterOffset - nLeftRightOffset]
         nVerticalNeighbourMapping32u = bestMappingNewA.ravel()[nCurIterOffset - nTopBotOffset]

         # Collect candidates
         # ## Type 1 ## 
         # Current pixel best mapping's entry in target image hash table
         nCurrTypeHashVal32u = pCodeA[j,i]
         for nHashTableCounter in range(nHashsNoe16u): 
            CurrBestError32u, CurrBestMapping32u = CalcSqrDifferenceAndTakeBest(CurrBestError32u, CurrBestMapping32u, hTables_B[nHashTableCounter][nCurrTypeHashVal32u], nCurIterOffset, pFiltA, pFiltB)
         # ## NEW - Type A ##
         nCurrTypeHashVal32u = pCodeA[j,i] # index into hash table
         for nHashTableCounter in range(nHashsNoe16u): 
            nCurrLocA32u = hTables_A[nHashTableCounter][nCurrTypeHashVal32u] # A-candidate
            nCurrCandMapping32u = bestMappingNewA.ravel()[nCurrLocA32u] # A index
            CurrBestError32u, CurrBestMapping32u = CalcSqrDifferenceAndTakeBest(CurrBestError32u, CurrBestMapping32u, nCurrCandMapping32u, nCurIterOffset, pFiltA, pFiltB)
         # Right of the left pixel best mapping's
         nCurrTypeMapping32u = nHorizNeighbourMapping32u + nLeftRightTargetOffset
         nCurrTypeHashVal32u = pCodeB.ravel()[nCurrTypeMapping32u]
         # ## Type 2 - lr ## 
         # Test for original candidate
         CurrBestError32u, CurrBestMapping32u = CalcSqrDifferenceAndTakeBest(CurrBestError32u, CurrBestMapping32u, nCurrTypeMapping32u, nCurIterOffset, pFiltA, pFiltB)
         # Test for cadidates in hash table with the same entry
         for nHashTableCounter in range(nHashsNoe16u): 
            nCurrCandMapping32u = hTables_B[nHashTableCounter,nCurrTypeHashVal32u]
            CurrBestError32u, CurrBestMapping32u = CalcSqrDifferenceAndTakeBest(CurrBestError32u, CurrBestMapping32u, nCurrCandMapping32u, nCurIterOffset, pFiltA, pFiltB)
                                    
         # ## Type 2 - tb ## 
         # Bottom of the top pixel best mapping's - Or otherwise
         nCurrTypeMapping32u = nVerticalNeighbourMapping32u + nTopBotOffset
         if ((nCurrTypeMapping32u>=0) and (nCurrTypeMapping32u<codeLen)):
            nCurrTypeHashVal32u = pCodeB.ravel()[nCurrTypeMapping32u]
            # Test for original candidate
            CurrBestError32u, CurrBestMapping32u = CalcSqrDifferenceAndTakeBest(CurrBestError32u, CurrBestMapping32u, nCurrTypeMapping32u, nCurIterOffset, pFiltA, pFiltB)
            # Test for cadidats in hash table with the same entry
            for nHashTableCounter in range(nHashsNoe16u): 
                  nCurrCandMapping32u = hTables_B[nHashTableCounter,nCurrTypeHashVal32u]
                  CurrBestError32u, CurrBestMapping32u = CalcSqrDifferenceAndTakeBest( CurrBestError32u, CurrBestMapping32u, nCurrCandMapping32u, nCurIterOffset, pFiltA, pFiltB)

         bestErrorsNewA[j,i] = CurrBestError32u
         bestMappingNewA[j,i] = CurrBestMapping32u

   return bestErrorsNewA, bestMappingNewA



def getCSHmap(frame, frame0, PATCH_WIDTH):

   pFiltA = GetResultsOfKernelApplication(frame, maxKernels, PATCH_WIDTH)
   pFiltB = GetResultsOfKernelApplication( frame0, maxKernels,  PATCH_WIDTH)

   w = frame.shape[1]
   h = frame.shape[0]
   w0 = frame0.shape[1]
	
   largeVal = np.int16(max(256*PATCH_WIDTH*PATCH_WIDTH-1, 32767))
   pFiltA, pFiltB = ModifyBoundaries(pFiltA, pFiltB, largeVal, PATCH_WIDTH)

   bitCountPerTable = np.uint(np.zeros(iters))
   indices_A_offset = w
   indices_B_offset = w
   hTables_A = np.zeros((numHashs, tableSize), dtype=np.uint32)
   hTables_B = np.zeros((numHashs, tableSize), dtype = np.uint32)
   for h1  in range(numHashs):
      for t in range(tableSize):
         hTables_A[h1,t] = np.uint32((indices_A_offset+2) + np.random.rand() * ( h*(w-1) - (indices_A_offset + 2) ) + 0.5)
         hTables_B[h1,t] = np.uint32((indices_B_offset+2) + np.random.rand() * ( h*(w-1) - (indices_B_offset + 2) ) + 0.5)		


   bestMappingNewA = np.zeros((h,w), dtype=np.uint32)
   coord = (1+ np.max([w0,w]) + PATCH_WIDTH + 1) - 1
   bestMappingNewA.fill(coord)

   bestErrorsNewA = np.zeros((h,w),dtype=np.uint32)
   bestErrorsNewA.fill(100000000000)
   
   pCodeA = np.zeros((h,w),dtype=np.uint32)
   pCodeB = np.zeros((h,w),dtype=np.uint32)
   for tableInd in range(iters):   
      pCodeA, pCodeB, bc = BuildWalshHadamardCodes(PATCH_WIDTH, pFiltA, pFiltB, iters, 3, pCodeA, pCodeB, bitCountPerTable[tableInd])
      bitCountPerTable[tableInd] = bc
#		// indices_A --> code_A     
      hTables_A = HashCodesIntoTables(h, w, maxBits,
			pCodeA, hTables_A, indices_A_offset, maxBits, tableSize, bitCountPerTable[tableInd])   
      hTables_B = HashCodesIntoTables(h, w, maxBits,
			pCodeB, hTables_B, indices_B_offset, maxBits, tableSize, bitCountPerTable[tableInd])    
      bestErrorsNewA, bestMappingNewA = GCKCodebookPropagation16s_FilterVectors( bestErrorsNewA, bestMappingNewA,  (tableInd%2) == 1, numHashs, pFiltA, pFiltB, pCodeA, pCodeB, hTables_A, hTables_B)  

   CSHnnMapX = np.zeros((h,w), dtype = int)
   CSHnnMapY = np.zeros((h,w), dtype = int)
   for r in range(h):
      for c in range(w):
         ind = bestMappingNewA[r,c]
         y = int(np.floor(ind/w))
         x = int(ind - y*w)
         CSHnnMapX[r,c] = x
         CSHnnMapY[r,c] = y
         
   return CSHnnMapX, CSHnnMapY