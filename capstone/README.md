The final production code for the Airbus Ship Detection project.

This project uses FastAPI to host a RESTful endpoint.  The expected input is an image file.  When the input is submitted via the API, it is then 
run through the Detectron2 predictor using the pre-built model.  The output from the predictor is then drawn directly onto the original image 
(if any ships are detected) and this output is returned to the submitter.

Steps to use:
1. Run code using uvicorn - "uvicorn app.app:app".  API defaults to 127.0.0.1:8000, but these settings can be configured.
2. Submit input image via the POST action.  Submit it as form data with the input file entered into the form as parameter "body".
