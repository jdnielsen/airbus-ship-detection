<h2>Airbus Ship Detection Challenge</h2>

<h3>Problem Statement</h3>
Ever-increasing shipping traffic over time has lead to increased chances of serious issues at sea, such as accidents and criminality.  Many parties such as envorionmental interest groups, insurance companies, and governments have an interest in being able to quickly find and track ships using satellite imagery.
<br /><br />
This is an interesting technological challenge because this project seeks to identify exactly where the ships are in the images, as opposed to simply classifying whether or not there is a ship present.  Expanding on this, the source images could include more than one ship in them, and the goal will be to detect all the ships in the images.  Additionally, the ships could potentially be obscured in the images by clouds or other obstructions.

<h3>Data</h3>
The data for this project consists of thousands of image files and a CSV file containing run-length encoded data that details the portions of the images that contain ships.
<br /><br />
The project concept and the data for it comes from a Kaggle challenge.

<h3>Problem Approach</h3>
This problem is a supervised classification problem.  I am attempting to predict, in an image, where the ships are in the image.  I will be using deep learning to solve this problem.

<h3>Deliverable</h3>
The deliverable for this project will be a docker image containing the API application.  The API application will accept image files, evaluate them using the trained model, and then return an image file with detected objects labeled on the image.  The docker image should be ready to deploy on any server that has docker installed and ready.

<h3>Computational Resources</h3>
For the training of the model, Google Colab Pro will be used with the High RAM and GPU options selected.
<br /><br />
For the final API, no great amount of computational resources will be necessary.  The API is configured to use CPU only.
