
IAGI - v8 2022-11-01 2:43pm
==============================

This dataset was exported via roboflow.com on November 1, 2022 at 2:44 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 78 images.
Grapes are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 50 percent of the image
* Random rotation of between -45 and +45 degrees
* Random shear of between -24° to +24° horizontally and -25° to +25° vertically


