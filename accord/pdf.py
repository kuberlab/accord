import pdfminer.settings

pdfminer.settings.STRICT = False
import pdfminer.high_level
import pdfminer.layout
import pdfminer.settings
from pdfminer.image import ImageWriter
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import io
import utils.lattice as lattice
import math


class COI(object):
    def __init__(self):
        self.Producer = ''
        self.ProducerConfidence = 0
        self.Insured = ''
        self.InsuredConfidence = 0
        self.Holder = ''
        self.HolderConfidence = 0
        self.Liability = []



def in_bbox(b, p, size):
    bx1 = b[0]
    by1 = b[1]
    bx2 = b[2] + size
    by2 = b[3] + 2
    return p[0] >= bx1 and p[0] <= bx2 and p[1] >= by1 and p[1] <= by2


def join_bbox(v1, v2):
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    return [
        min(v1[0], v2[0]),
        min(v1[1], v2[1]),
        max(v1[2], v2[2]),
        max(v1[3], v2[3]),
    ]


def interset(b1, b2, size):
    if b1 is None:
        return True
    if b2 is None:
        return True
    return in_bbox(b1, (b2[0], b2[1]), size) or in_bbox(b1, (b2[2], b2[1]), size) or in_bbox(b1, (b2[2], b2[3]),
                                                                                             size) or in_bbox(b1, (
    b2[0], b2[3]), size)


def extract_text(file, _py2_no_more_posargs=None,  # Bloody Python2 needs a shim
                 no_laparams=False, all_texts=True, detect_vertical=None,  # LAParams
                 word_margin=None, char_margin=None, line_margin=None, boxes_flow=None,  # LAParams
                 output_type='xml', codec='utf-8', strip_control=False,
                 maxpages=0, page_numbers=None, password="", scale=1.0, rotation=0,
                 layoutmode='normal', output_dir=None, debug=False,
                 disable_caching=False, **other):
    if _py2_no_more_posargs is not None:
        raise ValueError("Too many positional arguments passed.")

    # If any LAParams group arguments were passed, create an LAParams object and
    # populate with given args. Otherwise, set it to None.
    if not no_laparams:
        laparams = pdfminer.layout.LAParams()
        for param in ("all_texts", "detect_vertical", "word_margin", "char_margin", "line_margin", "boxes_flow"):
            paramv = locals().get(param, None)
            if paramv is not None:
                setattr(laparams, param, paramv)
    else:
        laparams = None

    imagewriter = None
    if output_dir:
        imagewriter = ImageWriter(output_dir)

    outfp = io.BytesIO()
    pdfminer.high_level.extract_text_to_fp(file, **locals())
    outfp.flush()
    return outfp.getvalue().decode('utf-8')


class Parser(object):
    def __init__(self, doc, draw=[]):
        self.src = doc
        self.show_img = None
        self.lines = []
        self.rects = []
        self.draw = draw
        self.original_img = None
        self.entries = []
        self.xml = ''
        self.latters = []
        self.curves = []

    def get_xml_bbox(self, v, h=0):
        bbox = v.split(',')
        bbox = [float(v) for v in bbox]
        if h > 0:
            bbox[1] = h - bbox[1]
            bbox[3] = h - bbox[3]
        if bbox[2] < bbox[0]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[3] < bbox[1]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        return bbox

    def parse(self):
        outfp = extract_text(self.src)
        self.xml = outfp
        root = ET.fromstring(outfp)
        self.lines = []
        self.rects = []
        page_bbox = None
        for page in root.findall('./page'):
            page_bbox = self.get_xml_bbox(page.attrib['bbox'])
            for textline in page.findall('./textbox/textline'):
                def _add_entry(latters, bbox):
                    text = ''.join(latters)
                    text = text.replace(u'\xa0', u' ')
                    text = text.strip()
                    if len(text) > 0:
                        self.entries.append((bbox, text))

            for textline in page.findall('./textbox/textline'):
                self.process_text(textline.findall('./text'), page_bbox, _add_entry)
            for textline in page.findall('./figure/textbox/textline'):
                self.process_text(textline.findall('./text'), page_bbox, _add_entry)
            for curve in page.findall('./curve'):
                pts = curve.attrib['pts'].split(',')

                for i in range(0,len(pts),2):
                    if (3+i)>=len(pts):
                        break
                    bbox = [float(pts[0+i]),page_bbox[3]-float(pts[1+i]),float(pts[2+i]),page_bbox[3]-float(pts[3+i])]

                    self.curves.append(bbox)
            for line in page.findall('./line'):
                width = int(line.attrib['linewidth'])
                bbox = self.get_xml_bbox(line.attrib['bbox'], page_bbox[3])
                self.lines.append(bbox)
                # img = cv2.line(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,0), thickness=width+1)
            for rect in page.findall('./rect'):
                width = int(rect.attrib['linewidth'])
                bbox = self.get_xml_bbox(rect.attrib['bbox'], page_bbox[3])
                #img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), width+1)
                self.rects.append(bbox)
            if page_bbox is not None:
                break

        img = np.ones((int(page_bbox[3]), int(page_bbox[2]), 3), np.uint8) * 255
        for bbox in self.lines:
            if (bbox[2] - bbox[0]) > 5 or (bbox[3] - bbox[1]) > 5:
                img = cv2.line(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 0), thickness=2)
        for bbox in self.rects:
            if bbox[2] - bbox[0] > 5 or (bbox[3] - bbox[1]) > 5:
                img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 0),
                                    thickness=2)

        self.original_img = img
        # self.show_img = np.ones((img.shape[0],img.shape[1],3),np.uint8)*255
        self.show_img = img.copy()
        tables, segments = lattice.extract_tables(img, v=15, h=30)
        for tn, cells in enumerate(tables):
            for rn, rows in enumerate(cells):
                for cn, c in enumerate(rows):
                    self.show_img = cv2.rectangle(self.show_img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])),
                                                  (255, 0, 0), 3)
                    self.show_img = cv2.putText(self.show_img, '{}: c={} r={}'.format(tn, cn, rn),
                                                (int(c[0]), int(c[1])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                                thickness=2, lineType=cv2.LINE_AA)
        coi = COI()
        for cells in tables:
            if len(cells) > 0:
                if len(cells[0]) > 0:
                    if len(cells[0]) < 5:
                        upper = cells[0][0][1]
                        down = cells[-1][0][1]
                        if upper < self.original_img.shape[0] / 5:
                            coi = self.table_one(coi, cells)
                        elif down > self.original_img.shape[0] / 2:
                            coi = self.table_three(coi, cells)
                    elif len(cells[0]) > 7:
                        coi = self.table_second(coi, cells)
        return coi, self.show_img

    def subtable(self, bbox, name, v=10, h=15, it=1):
        nimg = self.original_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        tables, _ = lattice.extract_tables(nimg, v=v, h=h, it=it)
        if name in self.draw:
            print('Draw {}'.format(name))
            self.show_img = cv2.rectangle(self.show_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                          (0, 0, 255), 3)
            for tn, cells in enumerate(tables):
                for rn, rows in enumerate(cells):
                    for cn, c in enumerate(rows):
                        c = (c[0] + int(bbox[0]), c[1] + int(bbox[1]), c[2] + int(bbox[0]), c[3] + int(bbox[1]))
                        self.show_img = cv2.rectangle(self.show_img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])),
                                                      (255, 0, 0), 3)
                        self.show_img = cv2.putText(self.show_img, '{}: c={} r={}'.format(tn, cn, rn),
                                                    (int(c[0]), int(c[1])),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                                    thickness=2, lineType=cv2.LINE_AA)
        else:
            print('Skip Draw {}/{}'.format(name, self.draw))
        if len(tables) > 0:
            cells = tables[0]
            for i, r in enumerate(cells):
                for j, c in enumerate(r):
                    b = cells[i][j]
                    cells[i][j] = (b[0] + int(bbox[0]), b[1] + int(bbox[1]), b[2] + int(bbox[0]), b[3] + int(bbox[1]))
            return cells
        return None

    def table_one(self, coi, cells):
        ncells = [[c[0]] for c in cells]
        ncells = self.subtable(self.get_bbox(ncells, 0, 1), 'first_table')
        if ncells is None or len(ncells) < 2:
            return coi

        def _producer_extract(t):
            t = t.lstrip('PRODUCER')
            return t.strip()

        def _insured_extract(t):
            t = t.lstrip('INSURED')
            return t.strip()

        prod_bbox = self.get_bbox([ncells[-2]], 0)
        ins_bbox = self.get_bbox([ncells[-1]], 0)

        prod, cprod = self.get_text(prod_bbox, text_map=_producer_extract)

        coi.Producer = prod
        coi.ProducerConfidence = cprod
        ins, cins = self.get_text(ins_bbox, text_map=_insured_extract)
        coi.Insured = ins
        coi.InsuredConfidence = cins
        return coi

    def table_second(self, coi, cells):
        if len(cells) < 4:
            return coi
        ncells = [c[-6:] for c in cells[2:]]
        ncells = self.subtable(self.get_bbox(ncells, 0), 'secod_table')
        cells = ncells
        # cells = cells[2:len(cells) - 1]
        if len(cells) < 5:
            return coi

        def _get_limits(i1, i2):
            amounts = []
            for i, row in enumerate(cells[i1:i2]):
                amount_bb = self.get_bbox([[row[-1]]], 0)

                def _amount(x):
                    try:
                        return float(x)
                    except:
                        return 0

                amount, camount = self.get_text(amount_bb, lambda c: c.isdigit(), _amount)
                name_bb = self.get_bbox([[row[-2]]], 0)
                tname, _ = self.get_text(name_bb)
                name_bb = self.get_bbox([[row[-2]]], 0)
                name, _ = self.get_text(name_bb)
                amounts.append({'name': name, 'value': amount, 'confidence': camount})
            return amounts

        def _policy_date(j, i1, i2):
            bb = self.get_bbox([[c[j]] for c in cells[i1:i2]], 0)
            return self.get_text(bb, lambda c: (c.isdigit() or c == ':' or c == '-' or c == '/'))

        def _policy_start_date(i1, i2):
            return _policy_date(-4, i1, i2)

        def _policy_end_date(i1, i2):
            return _policy_date(-3, i1, i2)

        def _policy_number(i1, i2):
            bb = self.get_bbox([[c[-5]] for c in cells[i1:i2]], 0)
            return self.get_text(bb, lambda c: (c.isalpha() or c.isdigit()) and c != ' ')

        def _policy_data(i1, i2):
            if len(cells) < i1:
                return None
            if len(cells) < i2:
                i2 = len(cells)
            n, cn = _policy_number(i1, i2)
            start, cstart = _policy_start_date(i1, i2)
            end, cend = _policy_end_date(i1, i2)
            limits = _get_limits(i1, i2)
            return {
                'policy': n,
                'policy_confidence': cn,
                'start': start,
                'start_confidence': cstart,
                'end': end,
                'end_confidence': cend,
                'limits': limits,
            }

        coi.Liability.append({'name': 'Commercial General Liability', 'data': _policy_data(0, 7)})
        coi.Liability.append({'name': 'Automobile Liability', 'data': _policy_data(7, 12)})
        coi.Liability.append({'name': 'Umbrela Liability', 'data': _policy_data(12, 15)})
        coi.Liability.append({'name': 'Worker Compensation', 'data': _policy_data(16, 19)})
        return coi

    def table_three(self, coi, cells):
        cells = [[c[0]] for c in cells]
        bb = self.get_bbox(cells, 0)
        holder, cholder = self.get_text(bb)
        coi.Holder = holder
        coi.HolderConfidence = cholder
        return coi

    def get_bbox(self, cells, skip, offset=0):
        minx = self.original_img.shape[1]
        maxx = 0
        miny = self.original_img.shape[0]
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
        minx = max(0, minx - offset)
        miny = max(0, miny - offset)
        maxx = min(self.original_img.shape[1], maxx + offset)
        maxy = min(self.original_img.shape[0], maxy + offset)
        return (minx, miny, maxx, maxy)

    def get_latters(self, bb, char_filter=None, text_map=None):
        l = []
        for e in self.latters:
            b = e[0]
            if self.is_in(bb, b):
                l.append(e[1])
        return l
        # return ""

    def get_text(self, bb, char_filter=None, text_map=None):
        text = []
        confidence = 100
        for e in self.entries:
            b = e[0]
            t = e[1]
            if self.is_in(bb, b):
                if char_filter is not None:
                    tmp = []
                    for c in t:
                        if char_filter(c):
                            tmp.append(c)
                    t = ''.join(tmp)
                if len(t) > 0:
                    text.append((b, t))

        text = sorted(text, key=lambda x: (x[0][3]))
        text = self.join_lines(text)
        if text_map is not None:
            text = text_map(text)
        return text, confidence

    def join_lines(self, tb):
        lines = []
        while len(tb) > 0:
            next = []
            line = []
            first = tb[0]
            median = first[0][1] + (first[0][3] - first[0][1]) / 2
            for b in tb:
                if b[0][1] < median and b[0][3] > median:
                    line.append(b)
                else:
                    next.append(b)
            line = sorted(line, key=lambda x: (x[0][0]))
            line = [e[1] for e in line]
            line = ' '.join(line)
            lines.append(line)
            tb = next
        return ' '.join(lines)

    def is_in(self, boxA, boxB, th=0.5):
        # print(boxB)
        xA = max(float(boxA[0]), boxB[0])
        yA = max(float(boxA[1]), boxB[1])
        xB = min(float(boxA[2]), boxB[2])
        yB = min(float(boxA[3]), boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / boxBArea >= th

    def process_text1(self, it, page_bbox, add_entry):
        # self.latters = []
        for t in it:
            l_bbox = self.get_xml_bbox(t.attrib['bbox'], page_bbox[3]) if 'bbox' in t.attrib else None
            if l_bbox is not None:
                self.latters.append((l_bbox, t.text))

    def process_text(self, it, page_bbox, add_entry):
        bbox = None
        size = 100
        latters = []
        added_count = 0
        for t in it:
            l_bbox = self.get_xml_bbox(t.attrib['bbox'], page_bbox[3]) if 'bbox' in t.attrib else None
            if l_bbox is not None:
                self.latters.append((l_bbox, t.text))

            # if l_bbox is not None:
            #    l_bbox[1], l_bbox[3] = page_bbox[3] - page_bbox[1] - l_bbox[3], page_bbox[3] - page_bbox[1] - l_bbox[1]
            l = t.text
            if len(l) > 0 and 'size' in t.attrib:
                size = min(size, math.floor((float(t.attrib['size']))))
            # l = l.strip()
            if len(l) > 0 and l != " ":
                if interset(bbox, l_bbox, size):
                    bbox = join_bbox(bbox, l_bbox)
                    latters.append(l)
                else:
                    if len(latters) > 0:
                        add_entry(latters, bbox)
                    latters = [l]
                    bbox = l_bbox
            else:
                if len(latters) > 0:
                    add_entry(latters, bbox)
                latters = []
                bbox = None
                size = 100
        if len(latters) > 0:
            add_entry(latters, bbox)
        return added_count
