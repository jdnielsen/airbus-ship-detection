<h2>Detecting Ships In Aerial Images</h2>

This repository is my capstone project for the Springboard Machine Learning Engineering track.  For this project, I have taken the <a href="https://www.kaggle.com/c/airbus-ship-detection">Airbus Ship Detection Challenge</a> on Kaggle as the basis of my project.  This provided me with several advantages - a real-world problem to solve with machine learning, interesting and unique challenges to overcome, and a large labeled dataset to work with.  My own project slightly differs from the Airbus challenge in that it performs object detection where the challenge was to perform image segmentation.

<h3>Data Collection</h3>
All data used in this project was acquired from the Kaggle challenge.  It consists of ten of thousands of aerial images formatted to be 768x768 in size, as well as a single CSV file that details segmentation of each individual ship that can be found in the training set of images.
<br /><br />
I decided to store the training data in my Google Drive storage, as I had intended to use Google Colab for training of the prototype.  Drive and Colab are integrated which made making the training data available to the machine learning software a simple matter.

<h3>Data Exploration.</h3>
Like I stated above, the data comes in two parts: many images and a CSV file detailing where the ships are in those images.  The images are self-explanatory, though the fact that they were all of uniform size was very convenient and spared me from having to make different sizes and shapes a consideration during the training phase.  The images themselves can be quite varied in appearance, with different environments seen in them from open water to beaches, canals, and harbors.  The water seen in the images can also be very different from one image to the next, with different colors and levels of choppiness making the apperance quite varied.  The images themselves also seem to have been taken at different altitudes and have varying levels of image quality.  Cloud coverage can be seen in some images, adding an additional variable into the mix.  These variations will be very helpful in training the model to generalize for different conditions.
<br /><br />
Here are some examples of the variations in the images:
<img src="https://drive.google.com/uc?export=view&id=1PancEx3XY3Vqp6-JH9faP98R0h-2cEB0" width="200" height="200" />
<img src="https://drive.google.com/uc?export=view&id=1bfGzFSyhaS0CECkTUxPIvVLKwBZEcnPb" width="200" height="200" />
<img src="https://drive.google.com/uc?export=view&id=1jVwyeT77yTrZ13XrHVB9Lxv2hsd56UXM" width="200" height="200" />
