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
    'nocr': 'tesseract',
}


def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    global PARAMS
    PARAMS.update(params)
    LOG.info("Init hooks")


def process(inputs, ctx):
    table_out = PARAMS['output'] == 'table'
    nfn = None
    if PARAMS['nocr']=='driver':
        import ocr.gunin as gunin
        def _fn(bbox,image):
            logging.info('CTX: {}'.format(ctx))
            n =  gunin.get_number(ctx.drivers[0],bbox,image)
            logging.info('From driver: {}'.format(n))
            return n
        nfn = _fn

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
            logging.info('Process page: {}'.format(i))
            img = np.array(p, np.uint8)
            img = img[:, :, ::-1]
            p = parse.Parser(img,draw=[],number_ocr=nfn)
            coi,show_img = p.parse()
            if parse.is_not_empty(coi):
                last_img = show_img
                if table_out:
                    result.append(coi)
                else:
                    result.append(coi.__dict__)
    else:
        logging.info('Process one document')
        p = parse.Parser(img,draw=[])
        coi,show_img = p.parse()
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
                    name = l['name']
                    l = l['data']
                    if l is not None:
                        table_output.append(
                            {'policy': name, 'policy_number': l['policy'], 'start': l['start'], 'end': l['end']})
                        for a in l.get('limits', []):
                            table_output.append({'limit': a['name'], 'value': str(int(a['value']))})
                    else:
                        table_output.append({'policy': name})
        table_meta = [
            {"name": "Producer", "label": "Producer"},
            {"name": "Insured", "label": "Insured"},
            {"name": "Holder", "label": "Holder"},
            {"name": "policy", "label": "Policy"},
            {"name": "policy_number", "label": "#Policy"},
            {"name": "start", "label": "Start"},
            {"name": "end", "label": "End"},
            {"name": "limit", "label": "Limit"},
            {"name": "value", "label": "Amount"},
        ]
        result = {
            'table_meta': json.dumps(table_meta),
            'table_output': json.dumps(table_output),
        }
        result['output'] = \
            cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8), params=[cv2.IMWRITE_JPEG_QUALITY, 95])[
                1].tostring()
        return result

    else:
        return {'docs': json.dumps(result)}
