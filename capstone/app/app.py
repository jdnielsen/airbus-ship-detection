from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
import io
import logging
import numpy as np
import predictor


logging.basicConfig(filename="app.log", level=logging.INFO)
logging.info("Application started.")
app = FastAPI()


@app.post("/detect_ships/")
def detect_ships(file: UploadFile = File(...)):
    """ Handle API command "detect_ships"

    The "detect_ships" API command accepts the input image, runs it through the detection function,
    and then returns a new image with the predicted objects drawn onto the original image.

    :param file: Image file that the predictor will analyze
    :return: A StreamingResponse that contains a jpeg image file of the model's analysis
    """
    logging.info("File (%s) submitted to detect_ships endpoint.",
                 file.filename if file.filename else "no file name provided")
    # check if the file is of appropriate type
    try:
        image_file = Image.open(file.file)
    except UnidentifiedImageError:
        logging.error("Unidentified image error.", exc_info=True)
        raise HTTPException(status_code=400, detail="Image file required.")
    # run file through predictor
    img = predictor.detect_ships(np.array(image_file))
    # put results into Image format
    img = Image.fromarray(img)
    # ensure image is in RGB format
    if not img.mode == "RGB":
        img = img.convert("RGB")
    # prepare results for streaming format
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    logging.info("Returning results to submitter.")

    return StreamingResponse(buffer, media_type="image/jpeg")
