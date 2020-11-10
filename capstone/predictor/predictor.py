import torch
import json
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

assert torch.__version__.startswith('1.6')


def detect_ships(img):
    """ Analyze input image using neural network model

    The detect_ships function runs the input image through the pre-trained NN model
    and returns the predicted output as an image with the detected objects drawn onto
    the original image.

    :param img: Input image for analysis
    :return: Image with model's analysis drawn onto original image
    """
    logging.info("Loading metadata information...")
    with open('./config/metadata.json') as json_file:
        metadata = json.load(json_file)

    logging.info("Generating Detectron2 configuration...")
    cfg_file = "./config/predictor_cfg.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = './models/model_30k.pth'

    logging.info("Making prediction on submitted image...")
    img_predictor = DefaultPredictor(cfg)
    prediction = img_predictor(img)

    logging.info("Generating prediction image...")
    v = Visualizer(img, metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(prediction['instances'])

    logging.info("Returning prediction image.")
    return out.get_image()
