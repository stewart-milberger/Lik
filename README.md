# Peter Lik vs the Average Joe

## Motivation
Photography is mainstream. Everyone is capable of taking a great photo, but what determines the photo’s quality? In such a competitive industry, I’m interested in what sets one apart from many. Peter Lik holds the record for the most expensive photo ever sold, but is his keen eye really worth $6 million for one photo? This analysis aims to identify characteristics or distinguishing features of his photos to set a world-class photographer apart from any average Joe.

## Goal
This project aims to build a convolutional neural net to classify an image as either a Peter Lik photograph or not. 


## Images
To obtain the Peter Lik(PL) images, lik.com was scraped using the python Requests, and BeautifulSoup libraries. A total of 728 images of varying resolutions were acquired. Considering the input shape of adata must be the same  for a CNN the most frequent resolution for the images was chosen(800x1600). 314 images were naturally that size, while there were 128 images which were 1600x1600, so these images were split horizontally to increase sample size. This procedure destroyed the original framing of the images(which is arguably one of the key features of Peter Lik photographs), but the same was done for non-Peter Lik photos to account for the offset. The PL images were then mirrored and flipped to increase the sample size to 2150.
The other class(non-PL) of image was obtained from a landscape image Kaggle dataset containing over 7000 images. After filtering out all images with low images, 1300 images remained. These images were then dynamically cropped to have the same(800x1600) resolution, then each image was inspected to ensure there were no watermarks, borders, or any otherwise unexpected features. Non-PL images were then mirrored to double the sample size, resultuing in 2600 images.


