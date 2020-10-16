Airbus Ship Detection Challenge

Problem Statement:
Ever-increasing shipping traffic over time has lead to increased chances of serious issues at sea, such as accidents and criminality.  Many parties such as envorionmental interest groups, insurance companies, and governments have an interest in being able to quickly find and track ships using satellite imagery.

This is an interesting technological challenge because this project seeks to identify exactly where the ships are in the images, as opposed to simply classifying whether or not there is a ship present.  Expanding on this, the source images could include more than one ship in them, and the goal will be to detect all the ships in the images.  Additionally, the ships could potentially be obscured in the images by clouds or other obstructions.

Data:
The data for this project consists of thousands of image files and a CSV file containing run-length encoded data that details the portions of the images that contain ships.

The project concept and the data for it comes from a Kaggle challenge.

Problem Approach:
This problem is a supervised classification problem.  I am attempting to predict, in an image, where the ships are in the image.  I will be using deep learning to solve this problem.

Deliverable:
I intend to make this a web application that takes a test image input and outputs to the page the predicted ship locations on the original image.  I may attempt to allow users to upload their own test images, but I think appropriate images for this project would be somewhat rare.  Therefore I think it more likely that I would host a gallery of the Kaggle challenges test images (ones that were not used in training the model) for users to select and see the results.

Computational Resources:
Not sure.  I will update this section.
