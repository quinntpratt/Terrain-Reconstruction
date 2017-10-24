# Terrain-Reconstruction
Surface generation from contour map images.

Using the OpenCV module this script allows the user to input an image (similar to the example provided) and reconstruct a surface (using the standard Delaunay Triangulation method - resulting in a piece-wise linear extrusion from irregularly-spaced data) from contour lines in the image.

This script could be easily modified to accommodate open contours with user-given elevation data.

The program contains a built-in module for closed, nested contours. 

Some user-input might be needed to adjust thresholds for proper contour-finding.

This program was developed to satisfy requirements for a MATH 380 (computational geometry) student project at the University of San Diego.
