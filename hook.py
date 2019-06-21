import cv2
import numpy as np
import logging
import pdf2image
import accord.parse as parse
import json

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    LOG.info("Init hooks")


def process(inputs, ctx):
    doc = inputs['doc'][0]
    img = None
    try:
        img = cv2.imdecode(np.frombuffer(doc, np.uint8), cv2.IMREAD_COLOR)
    except:
        None
    result = []
    if img is None:
        pages = pdf2image.convert_from_bytes(doc, 300)
        for i, p in enumerate(pages):
            img = np.array(p, np.uint8)
            img = img[:, :, ::-1]
            coi = parse.parse(img)
            if parse.is_not_empty(coi):
                result.append(coi.__dict__)
    else:
        coi = parse.parse(img)
        if parse.is_not_empty(coi):
            result.append(coi.__dict__)
    result = json.dumps(result, indent=2)
    return {'docs': result}
