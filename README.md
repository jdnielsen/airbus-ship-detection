<h2>Detecting Ships In Aerial Images</h2>

This repository is my capstone project for the Springboard Machine Learning Engineering track.  For this project, I have taken the <a href="https://www.kaggle.com/c/airbus-ship-detection">Airbus Ship Detection Challenge</a> on Kaggle as the basis of my project.  This provided me with several advantages - a real-world problem to solve with machine learning, interesting and unique challenges to overcome, and a large labeled dataset to work with.  My own project slightly differs from the Airbus challenge in that it performs object detection where the challenge was to perform image segmentation.

<h3>Data Collection</h3>
All data used in this project was acquired from the Kaggle challenge.  It consists of ten of thousands of aerial images formatted to be 768x768 in size, as well as a single CSV file that details segmentation of each individual ship that can be found in the training set of images.
<br /><br />
I decided to store the training data in my Google Drive storage, as I had intended to use Google Colab for training of the prototype.  Drive and Colab are integrated which made making the training data available to the machine learning software a simple matter.

<h3>Data Exploration</h3>
Like I stated above, the data comes in two parts: many images and a CSV file detailing where the ships are in those images.  The images are self-explanatory, though the fact that they were all of uniform size was very convenient and spared me from having to make different sizes and shapes a consideration during the training phase.  The images themselves can be quite varied in appearance, with different environments seen in them from open water to beaches, canals, and harbors.  The water seen in the images can also be very different from one image to the next, with different colors and levels of choppiness making the apperance quite varied.  The images themselves also seem to have been taken at different altitudes and have varying levels of image quality.  Cloud coverage can be seen in some images, adding an additional variable into the mix.  These variations will be very helpful in training the model to generalize for different conditions.
<br /><br />
Here are some examples of the variations in the images:
<img src="https://drive.google.com/uc?export=view&id=1PancEx3XY3Vqp6-JH9faP98R0h-2cEB0" width="300" height="300" />
<img src="https://drive.google.com/uc?export=view&id=1bfGzFSyhaS0CECkTUxPIvVLKwBZEcnPb" width="300" height="300" />
<img src="https://drive.google.com/uc?export=view&id=1jVwyeT77yTrZ13XrHVB9Lxv2hsd56UXM" width="300" height="300" />

<br />
The CSV file has two columns: the first column (ImageId) has the file name of the image, and the second column (EncodedPixels) has either a run length encoded string that describes what pixels in the image make up a single ship in that image or it has a value of NaN in the event that the image has no ships in it.  If there is a ship or multiple ships in an image, there will be a row in the CSV file for each ship in that image.
<br /><br />
In my exploration of the training dataset, I could see that of the >192k images there were 150k images that contained no ships at all.  Comparing this to the >230k rows in the CSV file, I could determine that there were >80k ships in >42k images that contained ships.

<h3>Preparing For Prototype Training</h3>
For this project I decided to use <a href="https://github.com/facebookresearch/detectron2">Facebook's Detectron2</a> to build my object detection model.  In my attempts to train a model with Detectron2, however, I did discover that the data I had would require some transformation in order to make it usable to the software.
<br /><br />
In order to train an object detection model, Detectron2 requires the coordinates of the bounding boxes of the objects to detect.  It also expects a single entry per image, as opposed to the multiple rows of the original CSV file.  Through some code, I took the CSV file and built out a JSON file which converted the run length encoded pixels from the original data and turned them into a list of bounding box coordinates.

<h3>Training The Prototype</h3>
With the data in a format usable by Detectron2, training became a fairly simple matter.  Some fiddling with the configuration was required to account for the data and the environment (Google Colab) but beyond that Detectron2 handled everything well.
<br /><br />
When scaling the prototype, one issue I discovered was that the sheer quantity of data (>192k images) could cause unpredictable behavior with Colab when it is all in one folder.  I resolved this issue by dividing up the image files into separate folders of no more than 1000 images per folder.  I had to adjust my JSON data file to also contain the new full path to the image file.  Aside from this, scaling was not an issue.
<br /><br />
Here is a graph of the AP (Average Precision) improving over the training period:
<img src="https://drive.google.com/uc?export=view&id=1-nCT7wSMwLrbKH5tEPYAXvjfsluXB4Wo" />
I had configured the trainer to save checkpoints every 2000 iterations.  Looking at the AP chart, it appears to level off at around 30 thousand iterations, so I took the checkpoint with 30 thousand iterations going forward.  Using this model, I can run images through the predictor to see how it does.  Here are some examples:
<br />
A basic example with two ships.
<img src="https://drive.google.com/uc?export=view&id=1MAK6hwGCGNZksQnj9y5oo8md4m-1ENUb" width="300" height="300" />
Smaller ships near land (Liberty Island!).
<img src="https://drive.google.com/uc?export=view&id=1zLYQrIKcfFR1lLm96RnRXX2S9eTpWnSW" width="300" height="300" />
Radically different water color, and the ship is only partly in the image.
<img src="https://drive.google.com/uc?export=view&id=1lxAixJ0torrB_cfUFecdqglwNt_rebW9" width="300" height="300" />
Took a slice of an image, to see how the predictor can handle a different size and shape image.
<img src="https://drive.google.com/uc?export=view&id=1pNpz9flPiFHLsAJ3T02itTvNr9Aob1UM" width="250" height="129" />
Surrounded by land.
<img src="https://drive.google.com/uc?export=view&id=1ZABRp3VbWSPU0FKYVEy11GwEwOPd-Nb7" width="300" height="300" />

<h3>Building The Application</h3>
Now that the model has been trained, I have completed the most challenging part of the project.  Next step is to make the model more accessible.  I did this by making a RESTful application that allows users to submit their images and receive in return the model's object detection predictions.
<br /><br />
There are two main components to the application: the API and the predictor.  The API has been built using the FastAPI library.  Using this library, it was very quick and simple to make an API endpoint.  This endpoint accepts POST queries and expects them to arrive as form data with the image file to be processed sent as a parameter named 'file'.  If these conditions are met, the API passes the image to the predictor code for it to process through the model.  That code should then return a new image to the API, which the API then converts into a file stream to return to the end user.
<br /><br />
The predictor portion of the code uses the Predictor and Visualizer modules from Detectron2 to make use of the trained model.  When an image is passed into the predictor, a Detectron2 DefaultPredictor is instantiated and configured to use the trained model as its weights.  The input image is passed through this DefaultPredictor to get the model's predictions.  These results are then fed into the Detectron2 Visualizer to have the predictions labelled directly onto the input image.  This new image is then returned to the API so that it can be provided to the end user.
<br /><br />
Once successfully deployed, this API can be used by, as stated earlier, submitting a POST command to the address {server IP address and port}/detect_ships.

<h3>Deployment</h3>
The final step for the project is to put it all together for easy deployment.  For this I built a docker image using 'python:3.6.12-slim-buster' as the base.  The following actions in the Dockerfile add in the various components and python modules that the application requires to run.  Ultimately, the image should expose port 8888 and then start up a uvicorn web server to host the API.  This docker image can be found on Docker Hub at jdnielsen/ship_detection_api.
<br /><br />
With the Docker image, deployment to any server or cloud service with Docker installed and running should be a simple matter.  I have successfully tested this on Paperspace.
