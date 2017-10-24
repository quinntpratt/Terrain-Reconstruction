#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 07:29:08 2017

@author: quinn
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import shapely.geometry as geom
import math as math
import matplotlib.patches as patches
from itertools import compress


def main():
    
    # Input image:
    im = cv2.imread('./IMG_1832.png') # good one.
    
    # convert the image to grayscale - this makes it easier to pull the contours.
    # this is only necessary if the input image is not already binary.
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # Turn the image into a mask using a threshold. This step often requires some guess-and-check
    ret,thresh = cv2.threshold(imgray,220,220,220,cv2.THRESH_BINARY)
    
    # Now we pass the image into our contour extraction function which primarily uses openCV
    
    # In this example the image we are using is already a binary image so we need not use the 
    # grayscale image from above.
    # so we set the image to pass as input to the grayscale image.
    im2pass = imgray
    
    contours, hierarchy = contourExtract(im2pass)
    
    # parameters for removing contours.
    length_thresh = 5
    position_thresh = 20
    
    # this routine removes the erroneous contours.
    contours_final, hierarchy_final = contourModify(contours, hierarchy,length_thresh,position_thresh)
    
    # This routine plots the contours.
    plotPointSet(contours_final)
    
    # determine the height to extrude contours to.
    heights = orderNested(contours_final)
    
    # build the surface with delaunay triangulation.
    constructSurface(contours_final,heights)
    
    
def contourExtract(im):
    
    plt.figure()
    plt.subplot(2, 2, 1)    
    plt.imshow(im,cmap='gray')
    plt.title('Input Contour Map for Reconstruction',fontsize=25)
    
    # Find contours routine
    im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_L1)
    # Draw the contours in red with a 2 pixel tickness
    img = cv2.drawContours(im, contours, -1, (255,0,0), 2)
    
    # Show the contours on top of the image. 
    plt.subplot(2, 2, 2)    
    plt.imshow(img)
    plt.title('OpenCV Discovered Edges',fontsize=25)
    
    return contours, hierarchy
    
    
def contourModify(contours, hierarchy,length_thresh,position_thresh):
    
    
    # Now we have to go through the painstaking process of editing down the number of points
    # to actually get good contours.
    
    # remove the extra dimension out of the hierarchy object.
    hierarchy = hierarchy[0,:,:]
    
    print('Set to delete contours of length < {}'.format(length_thresh))
    
    # first we'll delete all the contours that are too small to matter, and while we're at it
    # we will also delete the extra dimension from the contour arrays.
    contours_trimmed = []
    hierarchy_trimmed = []
    for i in range(len(contours)):
        cnt = contours[i]
        hier = hierarchy[i]
        if len(cnt) >= length_thresh:
            contours_trimmed.append(cnt[:,0,:])
            hierarchy_trimmed.append(hier)
        
    # we will use thresholds and comparative logic to find unique contours, and then 
    # only save the  unique ones for the final rendering.
     
    # we will save their path lengths, and their left-most point's position vector.        
    path_lengths = np.zeros((len(contours_trimmed)))
    leftmost_position = np.zeros((len(contours_trimmed)))
     
    for i in range(len(contours_trimmed)):
         cnt = contours_trimmed[i]
         x = cnt[:,0]
         y = cnt[:,1]
         n = len(x) 
         lv = [np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) for i in range (1,n)]
         path_lengths[i] = sum(lv)
         
         x_min_indx = np.argmin(x)
         leftmost_position[i] = np.sqrt((x[x_min_indx])**2 + (y[x_min_indx])**2)
    
      
    length_diff = abs(np.diff(path_lengths))
    leftmost_diff = abs(np.diff(leftmost_position))
    
    print('Difference between Path lengths: ')
    print(length_diff)
    print('Difference between Left-most Position : ')
    print(leftmost_diff)
    
    
    # for the remaining contours we will compare the differences in this left-most position vector
    # since duplicate contours have been observed to always be generated in close proximity in 
    # the contours list, we will compare the leftmost position vectors one after another.
    print('-------Uniqueness Loop-------')
    
    # absolute difference position vector threshold.
    position_threshold = 30
    # create a copy of our list of contours, we will be deleting things out of the list...
    contour_list = contours_trimmed
    hierarchy_list = hierarchy_trimmed
    i = 0
    while i < len(contour_list):
        cnt = contour_list[i]
        hier = hierarchy_list[i]
        
        path = path_lengths[i]
        leftmost = leftmost_position[i]
            
        # compute both the absolute difference in position and the absolute difference
        # in path length.
        length2compare = abs(path_lengths - path)
        position2compare = abs(leftmost_position - leftmost)
        
        # remove those which have the same length, i.e. itself
        bool_mask = length2compare > 0
        length2compare = length2compare[bool_mask]
        position2compare = position2compare[bool_mask]
        
        
        # find the median path length difference, use this for another mask.
        med = np.median(sorted(length2compare))
        
        bool_mask = length2compare < med*0.75
        length2compare = length2compare[bool_mask]
        position2compare = position2compare[bool_mask]
        
        # now we have a list containing the absolute difference in the position vectors 
        # for the contours whose path lengths are less than the median path length.
        
        if any(position2compare < position_threshold):
            print('deleteing contour {}'.format(i))
            del contour_list[i]
            del hierarchy_list[i]
            
        i = i + 1
            
    contours_final = contour_list
    hierarchy_final = hierarchy_list
    print('-----------------------------')  
    print('Final Contours Extracted = {}'.format(len(contours_final)))
    
    return contours_final, hierarchy_final
    
def plotPointSet(contours_final):
    
    # Now we plot all of the contours! 
    ds = 2
    plt.subplot(2, 2, 3) 
    for c in range(len(contours_final)):
        #extracts a particular array in the list of contours.
        cntrs = contours_final[c]    
        plt.plot(cntrs[::ds,0],cntrs[::ds,1],'b.',ms=15)
                
    plt.gca().invert_yaxis()
    plt.grid()
    plt.title('Point Reconstruction from Edges',FontSize=24)
    
def orderNested(contours_final):
    
    # for each contour we form a polygon class.
    polygons = [0] * len(contours_final)
    for i in range(len(contours_final)):
        
        cnt = contours_final[i]
        
        verts = list([tuple(row) for row in cnt])
        # compute centroid
        cent=(sum([v[0] for v in verts])/len(verts),sum([v[1] for v in verts])/len(verts))
        # sort by polar angle
        verts.sort(key=lambda v: math.atan2(v[1]-cent[1],v[0]-cent[0]))
        
        polygons[i] = geom.polygon.Polygon(verts)
        
        # plot points
        ax = plt.subplot(2,2,4)
        plt.scatter([v[0::4] for v in verts],[v[1::4] for v in verts])
        # plot polyline
        plt.gca().add_patch(patches.Polygon(verts,closed=True,fill=True,
               color=tuple(np.random.random(3)),alpha=0.5))
        ax.invert_yaxis()
        plt.title('Polygon Constructon from Contours',fontsize=24)

    # now we loop through and check for nested-ness
    heights = [0] * len(contours_final)
    for p in range(len(polygons)):
        mask = [True] * len(contours_final)
        mask[p] = False
        # pull the polygon of interest.
        poly = polygons[p]
        # other polygons to search through:
        poly_2_check = list(compress(polygons, mask))
    
        for k in range(len(poly_2_check)):
            poly2 = poly_2_check[k]    
            point_list = [geom.point.Point(pnt) for pnt in poly2.exterior.coords]
            
            nested_bool = all([poly.contains(pnt) for pnt in point_list])
        
            if nested_bool:
                heights[k+1] = heights[k+1] + 1
    
    print(heights) 
    return heights
    
    
def constructSurface(contours_final,heights):
    # Now that we have our contours extracted it's time to assign height values to them.
    # for starters we'll just do this manually.

    cnt = contours_final[0]
    z = np.ones((len(cnt),1))*heights[0]
    point_set = np.hstack((cnt,z))

    for i in range(1,len(contours_final)):
        cnt = contours_final[i]
        z = np.ones((len(cnt),1))*heights[i]
        xyz = np.hstack((cnt,z))
        point_set = np.vstack((point_set,xyz))
    
    print('Total Points to be Considered = {}'.format(len(point_set)))

    plt.figure()
    #ls = LightSource(azdeg=315, altdeg=45)
    ax = plt.gca(projection='3d')
    surf = ax.plot_trisurf(point_set[:,0], point_set[:,1], point_set[:,2],
                           cmap=cm.terrain, alpha = 0.75,
                           linewidth=0.2, antialiased=True, shade=True)

    plt.title('Surface Reconstructed from Contour Map',fontsize=24)
    plt.gca().invert_yaxis()

    
    
if __name__ == '__main__':
    main()