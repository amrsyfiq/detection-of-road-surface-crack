# Detection-of-Road-Surface-Crack

This project aims at the problem of road surface image denoising and crack recognition by using embedded camera. Using Gaussian filter to blur the image, over the threshold zero processing and the morphology on and off operation are binarization and further denoising. And the crack contour is marked by FAST feature point recognition, which reach for different road crack damage can be identified. There are two parts to the innovation: First, using a lower-cost embedded camera to capture photos, and second, using the characteristics of large difference in gray value between crack region and other regions, the crack profile is marked by the way of FAST feature point recognition. It makes the identification system more integrated and portable, and can judge the crack more accurately and reduce the cost.

## Requirement
1.  opencv
2.  numpy
3.  scikit-image
4.  matplotlib
