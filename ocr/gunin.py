import numpy as np
import logging
import cv2

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', ':', '(', ')', '.', ',', '/'
    # Apostrophe only for specific cases (eg. : O'clock)
                                  "'",
    " ",
    # "end of sentence" character for CTC algorithm
    '_'
]


def read_charset():
    charset = {}
    inv_charset = {}
    for i, v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset


chrset_index = read_charset()


def get_number_fn(model_path):
    from ml_serving.drivers import driver
    drv = driver.load_driver('tensorflow')
    serving = drv()
    serving.load_model(model_path)

    def _fn(bbox, img):
        return get_number(serving, bbox, img)

    return _fn

count = 0

def get_number(drv, bbox, img):
    minx = min(bbox[0]+2,img.shape[1])
    maxx = max(bbox[2]-2,0)
    miny = min(bbox[1]+2,img.shape[0])
    maxy = max(bbox[3]-2,0)
    image = img[miny:maxy,minx:maxx, ::-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (320, 32),interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    global count
    count += 1
    print("numbers/amout-{}.jpg".format(count))
    cv2.imwrite("numbers/amout-{}.jpg".format(count),image)
    image = np.stack([image, image, image], axis=-1)
    image = image.astype(np.float32) / 127.5 - 1
    outputs = drv.predict({'images': np.stack([image], axis=0)})
    predictions = outputs['text']
    confidence = outputs['confidence']
    line = []
    end_line = len(chrset_index) - 1
    for i in predictions[0]:
        if i == end_line:
            break
        t = chrset_index.get(i, -1)
        if t == -1:
            continue
        if t.isdigit():
            line.append(t)
    try:
        return int(''.join(line)),int(confidence[0]*100)
    except:
        return 0,0
