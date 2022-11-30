
IAGI - v20 2022-11-06 1:24pm
==============================

This dataset was exported via roboflow.com on November 6, 2022 at 12:25 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 87 images.
Grapes are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 50 percent of the image
* Random rotation of between -20 and +20 degrees
* Random brigthness adjustment of between -55 and +55 percent


