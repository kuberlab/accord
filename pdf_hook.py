import cv2
import numpy as np
import logging
import accord.pdf as pdf
import json
import io

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def init_hook(**params):
    LOG.info("Init hooks")


def process(inputs, ctx):
    doc = inputs['doc'][0]
    p = pdf.Parser(doc=io.BytesIO(doc), draw=[])
    coi, _ = p.parse()
    table_output = []
    for r in [coi]:
        def _text(v, c):
            return '{} ({})'.format(v, c)

        table_output.append({'Producer': _text(r.Producer, r.ProducerConfidence),
                             'Insured': _text(r.Insured, r.InsuredConfidence),
                             'Holder': _text(r.Holder, r.HolderConfidence)})
        if r.Liability is not None:
            for l in r.Liability:
                name = l['name']
                l = l['data']
                if l is not None:
                    table_output.append(
                        {'policy': name, 'policy_number': _text(l['policy'], l['policy_confidence']),
                         'start': _text(l['start'], l['start_confidence']),
                         'end': _text(l['end'], l['end_confidence'])})
                    for a in l.get('limits', []):
                        table_output.append({'limit': a['name'], 'value': _text(a['value'], a['confidence'])})
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
            {"name": "value", "label": "Amount"}
        ]
        result = {
            'table_meta': json.dumps(table_meta),
            'table_output': json.dumps(table_output),
        }
        result['output'] = \
            cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8), params=[cv2.IMWRITE_JPEG_QUALITY, 95])[
                1].tostring()
        return result
