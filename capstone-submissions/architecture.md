<h2>Ship Detection Architecture</h3>

The final product of this project is an API that accepts an input image and returns that same image with object predictions 
drawn unto it.

<h3>Components:</h3>
The two major components are the API code and the prediction code.  The API accepts and verifies the input.  The input is 
then run through the predictor, which uses the pre-built model to make object detection predictions.  The output from the 
predictor is then returned to the API, which delivers the output to the end user.

<h3>Data:</h3>
The only data that is used in this API is the user's input image, which is not stored.  The model used for object detection 
has been pre-trained using thousands of images, but they are not necessary for the final API.

<h3>ML Model Lifecycle</h3>
This project is intended to detect ships in overhead/aerial images.  I would not expect the appearance of ships from an 
overhead view to change very much in the near future, so I don't think a retraining of the model is necessary on any 
particular interval.  Of course, additional data or improvements in the object detection training software could make 
retraining a good idea.

In the event that a renewed training of the model is desired, a similar approach to what I took earlier (see my prototype in 
the notebooks folder) would likely be used.  The original data (thousands of image files accompanied by data detailing the 
locations of all ships in the image), along with any new data that is available, would be collected and organized on the 
machine that will accomplish the training.  The training produces a .PTH model file as its only artifact.  This can be 
deployed to this API by simply swapping in the new model file and ensuring that the software is pointing to the new 
file name and location as its model.

<h3>Monitoring</h3>
The API and predictor software have logging enabled at key points in the software, so the first step in debugging will be to 
check the logs.

<h3>Tools/Technologies</h3>
<ul>
<li>Detectron2 - Machine Learning library used for training the model initially and for performing predictions through
the model</li>
<li>FastAPI - Library used for constructing the API</li>
<li>Uvicorn - Server software hosting the API</li>
</ul>

<h3>Costs</h3>
In reference to just implementing the final product of an API running input images through the pre-built model and returning
predictions to the end user, implementation costs are minimal.  A docker image containing the API and prediction code has 
been developed and is ready to be deployed, so it should be as simple as deploying the image to a cloud service.  Hardware 
requirements are minimal as the software is assuming no GPU access.
