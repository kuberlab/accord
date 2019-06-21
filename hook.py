import cv2
import numpy as np
import logging
import pdf2image
import accord.parse as parse
import json

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

PARAMS = {
    'output': 'json',
}


def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    global PARAMS
    PARAMS.update(params)
    LOG.info("Init hooks")


def process(inputs, ctx):
    table_out = PARAMS['output'] == 'table'
    doc = inputs['doc'][0]
    img = None
    last_img = None
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
            coi, show_img = parse.parse(img)
            if parse.is_not_empty(coi):
                last_img = show_img
                if table_out:
                    result.append(coi)
                else:
                    result.append(coi.__dict__)
    else:
        coi, show_img = parse.parse(img)
        if parse.is_not_empty(coi):
            last_img = show_img
            if table_out:
                result.append(coi)
            else:
                result.append(coi.__dict__)
    if table_out:
        table_output = []
        for r in result:
            table_output.append({'Producer': r.Producer, 'Insured': r.Insured, 'Holder': r.Holder})
            if r.Liability is not None:
                for l in r.Liability:
                    if l is not None:
                        table_out.append({'policy': l['policy'], 'start': l['start'], 'end': l['end']})
                        for a in l.get('limits', []):
                            table_out.append({'limit': a['name'], 'value': l['value']})
        table_meta = [
            {"name": "Producer", "label": "Producer"},
            {"name": "Insured", "label": "Insured"},
            {"name": "Holder", "label": "Holder"},
            {"name": "policy", "label": "#Policy"},
            {"name": "start", "label": "Start"},
            {"name": "end", "label": "End"},
            {"name": "limit", "label": "Limit"},
            {"name": "value", "label": "Amount", "type": "number", "format": ".1f"},
        ]
        result = {
            'table_meta': json.dumps(table_meta),
            'table_output': json.dumps(table_output),
        }
        if last_img is not None:
            result['output'] = cv2.imencode(".jpg", last_img, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()
        return result

    else:
        return {'docs': json.dumps(result)}
