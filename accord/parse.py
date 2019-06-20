import utils.lattice as lattice
import cv2
import numpy as np
import pytesseract
import PIL.Image as Image


def parse(img):
    tables, segments = lattice.extract_tables(img)
    show_img = np.copy(img)
    for tn, cells in enumerate(tables):
        for rn, rows in enumerate(cells):
            for cn, c in enumerate(rows):
                show_img = cv2.rectangle(show_img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 0), 3)
                show_img = cv2.putText(show_img, '{}: c={} r={}'.format(tn, cn, rn), (int(c[0]), int(c[1])),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                       thickness=2, lineType=cv2.LINE_AA)
    vertical_segments, horizontal_segments = segments
    for l in vertical_segments:
        img = cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), thickness=5)
    for l in horizontal_segments:
        img = cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), thickness=5)

    for cells in tables:
        # print('{}-{}'.format(len(cells), len(cells[0])))
        if len(cells) > 0:
            if len(cells[0]) > 0:
                if len(cells[0]) < 4:
                    upper = cells[0][0][1]
                    down = cells[-1][0][1]
                    if upper < img.shape[0] / 5:
                        # First table
                        # print("First-{}".format(get_bbox(cells,img,3)))
                        table_one(img, cells)
                    elif down > img.shape[0] / 2:
                        # print("Third-{}".format(get_bbox(cells,img,0)))
                        table_three(img,cells)
                elif len(cells[0]) > 9:
                    table_second(img, cells)
                    # print("Second-{}".format(get_bbox(cells,img,2)))


def table_one(img, cells):
    bb = get_bbox(cells, img, 2)
    data = extact_text(img, bb)
    prod_bbox = get_bbox([[c[0]] for c in cells[4:7]], img, 0)
    prod = get_text(prod_bbox, data)
    print('Prod: {}'.format(prod))
    ins_bbox = get_bbox([[c[0]] for c in cells[8:]], img, 0)
    ins = get_text(ins_bbox, data)
    print('Ins: {}'.format(ins))

def table_three(img, cells):
    cells = [[c[0]] for c in cells]
    bb = get_bbox(cells, img,0)
    data = extact_text(img, bb)
    holder = get_text(bb, data)
    print('Holder: {}'.format(holder))

def table_second(img, cells):
    if len(cells) < 4:
        return
    cells = cells[2:len(cells) - 1]
    bb = get_bbox(cells, img, 0)
    data = extact_text(img, bb)

    def _get_limits(i1, i2, names):
        amounts = {}
        for i, row in enumerate(cells[i1:i2]):
            amount_bb = get_bbox([[row[-1]]], img, 0)

            def _amount(x):
                try:
                    return float(x)
                except:
                    return 0

            amount = get_text(amount_bb, data, lambda c: c.isdigit(), _amount)
            name = names.get(i, '')
            if name == '':
                name_bb = get_bbox([[row[-2]]], img, 0)
                name = get_text(name_bb, data)
            if name != '':
                amounts[name] = amount
        return amounts

    def _policy_date(j, i1, i2):
        bb = get_bbox([[c[j]] for c in cells[i1:i2]], img, 0)
        return get_text(bb, data, lambda c: (c.isdigit() or c == ':' or c == '-' or c == '/'))

    def _policy_start_date(i1, i2):
        return _policy_date(-4, i1, i2)

    def _policy_end_date(i1, i2):
        return _policy_date(-3, i1, i2)

    def _policy_number(i1, i2):
        bb = get_bbox([[c[-5]] for c in cells[i1:i2]], img, 0)
        return get_text(bb, data, lambda c: (c.isalpha() or c.isdigit()) and c != ' ')

    def _policy_data(i1, i2, names={}):
        if len(cells) < i1:
            return None
        if len(cells) < i2:
            i2 = len(cells)
        n = _policy_number(i1, i2)
        start = _policy_start_date(i1, i2)
        end = _policy_end_date(i1, i2)
        limits = _get_limits(i1, i2, names)
        return n, start, end, limits

    general = {0: 'EACH OCCURRENCE', 1: 'DAMAGE TO RENTED PREMISES (Ea Occurrence)', 2: 'MED EXP (Anyoneperson)',
               3: 'PERSONAL & ADV INJURY', 4: 'GENERAL AGGREGATE', 5: 'PRODUCTS - COMP/OP AGG'}
    auto = {0: 'BODILY INJURY (Per person)', 1: 'BODILY INJURY (Per accident)', 2: 'PROPERTY DAMAGE (Per accident)'}
    umbrela = {0: 'EACH OCCURRENCE', 1: 'AGGREGATE'}
    worker = {0: 'STATUTE OTHER', 1: 'E.L. EACH ACCIDENT', 2: 'E.L. DISEASE - EA EMPLOYEE',
              3: 'E.L. DISEASE - POLICY LIMIT'}
    print('Commercial General Liability: {}'.format(_policy_data(0, 7, general)))
    print('Automobile Liability: {}'.format(_policy_data(8, 12, auto)))
    print('Umbrela Liability: {}'.format(_policy_data(12, 15, umbrela)))
    print('Worker Compensation: {}'.format(_policy_data(15, 19, worker)))


def get_text(bb, data, char_filter=None, text_map=None):
    text = []
    for e in data:
        b = e['bbox']
        if is_in(bb, b):
            t = e['text']
            if char_filter is not None:
                tmp = []
                for c in t:
                    if char_filter(c):
                        tmp.append(c)
                t = ''.join(tmp)
            text.append((b, t))
    text = sorted(text, key=lambda x: (x[0][1], x[0][0]))
    text = [e[1] for e in text]
    text = ' '.join(text)
    if text_map is not None:
        text = text_map(text)
    return text


def extact_text(img, bbox):
    to_process = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    data = pytesseract.image_to_data(Image.fromarray(to_process), config='', output_type=pytesseract.Output.DICT)
    entry = []
    for i, text in enumerate(data['text']):
        if text is None:
            continue
        text = text.replace('|', '')
        text = text.replace('_', '')
        if text.strip() == '':
            continue
        left = data['left'][i] + bbox[0]
        top = data['top'][i] + bbox[1]
        width = data['width'][i]
        height = data['height'][i]
        entry.append({'bbox': [left, top, left + width, top + height], 'text': text})
    return entry


def is_in(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / boxBArea > 0.5


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    d = float(boxAArea + boxBArea - interArea)
    if d == 0:
        return 0
    iou = interArea / d
    return iou


def get_bbox(cells, img, skip):
    minx = img.shape[1]
    maxx = 0
    miny = img.shape[0]
    maxy = 0
    if skip > 0 and len(cells) > skip:
        cells = cells[skip:]
    for rows in cells:
        for col in rows:
            if int(col[0]) < minx:
                minx = int(col[0])
            if int(col[2]) > maxx:
                maxx = int(col[2])
            if int(col[1]) < miny:
                miny = int(col[1])
            if int(col[3]) > maxy:
                maxy = int(col[3])
    return (minx, miny, maxx, maxy)
