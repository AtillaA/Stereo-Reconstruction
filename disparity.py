import numpy as np
import cv2

def BlockMatching(left, right, maxDisparity = 48, window_size = 11):
    limg=np.asanyarray(left,dtype=np.double)
    rimg=np.asanyarray(right,dtype=np.double)
    img_size=np.shape(limg)[0:2]

    # Accelerated SAD algorithm.

    imgDiff=np.zeros((img_size[0],img_size[1],maxDisparity))
    e = np.zeros(img_size)
    for i in range(0,maxDisparity):
        e=np.abs(rimg[:,0:(img_size[1]-i)]- limg[:,i:img_size[1]]) 
        e2=np.zeros(img_size)

    # This part of the integrated matrix is ​​calculated, according to the size of the window, to get the same position, the grayscale difference of different parallaxes

        for x in range((window_size),(img_size[0]-window_size)):
            for y in range((window_size),(img_size[0]-window_size)):
                e2[x,y]=np.sum(e[(x-window_size):(x+window_size),(y-window_size):(y+window_size)])
        imgDiff[:,:,i]=e2
    dispMap=np.zeros(img_size)

    # This part finds the parallax that minimizes the grayscale difference, and draws

    for x in range(0,img_size[0]):
        for y in range(0,img_size[1]):
            val=np.sort(imgDiff[x,y,:])
            if np.abs(val[0]-val[1])>10:
                val_id=np.argsort(imgDiff[x,y,:])
                dispMap[x,y]=val_id[0]/maxDisparity*255
    #plt.imshow(dispMap)
    #plt.show()
    return dispMap

def BlockMatching2(left, right, maxDisparity = 48, window_size = 11):
    rows, cols = left.shape

    kernel = np.ones([window_size, window_size]) / window_size

    disparity_maps = np.zeros(
        [left.shape[0], left.shape[1], maxDisparity])
    for d in range(0, maxDisparity):
        # shift image
        translation_matrix = np.float32([[1, 0, d], [0, 1, 0]])
        shifted_image = cv2.warpAffine(
            right, translation_matrix,
            (right.shape[1], right.shape[0]))
        # calculate squared differences
        SAD = abs(np.float32(left) - np.float32(shifted_image))
        # convolve with kernel and find SAD at each point
        filtered_image = cv2.filter2D(SAD, -1, kernel)
        disparity_maps[:, :, d] = filtered_image

    disparity = np.argmin(disparity_maps, axis=2)
    disparity = np.uint8(disparity * 255 / maxDisparity)
    disparity = cv2.equalizeHist(disparity)
    return disparity

def ComputeDisparity(img1, img2, numDisparities=50, blockSize=21):
    h, w = img1.shape
    DbasicSubpixel = np.zeros(img1.shape)
    half = blockSize // 2
    #print('Half: ',half)
    for m in range(h):
        minr = max(0, m - half)
        maxr = min(h - 1, m + half + 1)
        for n in range(w):
            minc = max(0, n - half)
            maxc = min(w - 1, n + half + 1)

            mind = 0
            maxd = min(numDisparities, w - maxc)

            template = img2[minr:maxr, minc:maxc]

            numBlocks = maxd - mind + 1

            blockDiffs = np.zeros(numBlocks)

            for i in range(mind,maxd+1):

                block = img1[minr:maxr, (minc+i):(maxc+i)]

                blockIndex = i - mind

                blockDiffs[blockIndex] = np.sum(np.absolute(template - block))         

            bestMatchIndex = np.argmax(blockDiffs)
            d = bestMatchIndex + mind

            if (bestMatchIndex == 0) or (bestMatchIndex == numBlocks - 1):
                DbasicSubpixel[m, n] = d
            else:
                C1 = blockDiffs[bestMatchIndex - 1]
                C2 = blockDiffs[bestMatchIndex]
                C3 = blockDiffs[bestMatchIndex + 1]
                DbasicSubpixel[m, n] = d# - (0.5 * (C3 - C1) / (C1 - (2*C2) + C3))
                
    return DbasicSubpixel


def DepthMapBuiltIn(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg