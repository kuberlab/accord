import utils.lattice as lattice
import cv2
import numpy as np
import pytesseract
import PIL.Image as Image
import utils.deroted as deroted
import Levenshtein


class COI(object):
    def __init__(self):
        self.Producer = ''
        self.Insured = ''
        self.Holder = ''
        self.Liability = []



def is_not_empty(coi):
    return coi.Producer != '' or coi.Insured != '' or coi.Holder != '' or len(coi.Liability) > 0

class Parser(object):
    def __init__(self,img,draw=[]):
        img, _ = deroted.derotate3(img)
        self.original_img = img
        self.show_img = np.copy(img)
        self.parse_img = np.copy(img)
        self.draw = draw

    def parse(self):
        tables, segments = lattice.extract_tables(self.original_img,v=15,h=30)
        if 'main_table' in self.draw:
            for tn, cells in enumerate(tables):
                for rn, rows in enumerate(cells):
                    for cn, c in enumerate(rows):
                        self.show_img = cv2.rectangle(self.show_img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 0), 3)
                        self.show_img = cv2.putText(self.show_img, '{}: c={} r={}'.format(tn, cn, rn), (int(c[0]), int(c[1])),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                           thickness=2, lineType=cv2.LINE_AA)
        vertical_segments, horizontal_segments = segments
        for l in vertical_segments:
            self.parse_img = cv2.line(self.parse_img, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), thickness=3)
        for l in horizontal_segments:
            self.parse_img = cv2.line(self.parse_img, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), thickness=3)

        coi = COI()
        for cells in tables:
            # print('{}-{}'.format(len(cells), len(cells[0])))
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


    def remove_prefix(self,prefix, t):
        words = t.split(' ')
        if len(words) < 1:
            return t
        if Levenshtein.ratio(words[0].lower(), prefix.lower()) > 0.75:
            if len(words) > 1:
                print('Skip: {}'.format(words[0]))
                return ' '.join(words[1:])
            return ''
        else:
            return t


    def table_one(self,coi, cells):
        ncells = [[c[0]] for c in cells]
        ncells = self.subtable(self.get_bbox(ncells,0,10),'first_table',it=3)
        if ncells is not None:
            print('one use subtables')
            cells = ncells
            if len(cells)>2:
                cells = cells[-2:]
            bb = self.get_bbox(cells,0)
            data = self.extact_text(bb)
            prod_bbox = self.get_bbox([c for c in cells[-2:-1]], 0)
            ins_bbox = self.get_bbox([c for c in cells[-1:]], 0)
        else:
            bb = self.get_bbox(cells, 2)
            data = self.extact_text(bb)
            prod_bbox = self.get_bbox([[c[0]] for c in cells[3:7]], 0)
            ins_bbox = self.get_bbox([[c[0]] for c in cells[8:]], 0)

        def _producer_extract(t):
            return self.remove_prefix('Producer', t)

        def _insured_extract(t):
            return self.remove_prefix('INSURED', t)

        prod = self.get_text(prod_bbox, data, text_map=_producer_extract)
        show_img = cv2.rectangle(self.show_img, (int(prod_bbox[0]), int(prod_bbox[1])), (int(prod_bbox[2]), int(prod_bbox[3])), (0, 0, 255), 2)
        coi.Producer = prod
        ins = self.get_text(ins_bbox, data, text_map=_insured_extract)
        coi.Insured = ins
        self.show_img = cv2.rectangle(show_img, (int(ins_bbox[0]), int(ins_bbox[1])), (int(ins_bbox[2]), int(ins_bbox[3])), (0, 0, 255), 2)
        return coi


    def table_three(self,coi, cells):
        cells = [[c[0]] for c in cells]
        bb = self.get_bbox(cells, 0)
        self.show_img = cv2.rectangle(self.show_img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
        data = self.extact_text(bb)
        holder = self.get_text(bb, data)
        coi.Holder = holder
        return coi


    def join_cells(self,cells):
        hs = []
        for row in cells:
            hs.append(-1*self.get_row_height(row))
        hs = sorted(hs)
        h = hs[-1]
        if len(hs)>20:
            h = hs[20]*0.9
        return self.join_rows(cells,h)

    def get_row_height(self,row):
        h = None
        for c in row:
            if h is None:
                h = (c[3]-c[1])
                continue
            if (c[3]-c[1])<h:
                h = (c[3]-c[1])
        return h
    def join_rows(self,cells,min_height):
        cells = list(filter(lambda row: self.get_row_height(row)>=min_height,cells))
        return cells


    def subtable(self,bbox,name,v=10,h=15,it=2):
        nimg = self.original_img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
        tables, _ = lattice.extract_tables(nimg,v=v,h=h,it=it)
        if name in self.draw:
            print('Draw {}'.format(name))
            self.show_img = cv2.rectangle(self.show_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
            for tn, cells in enumerate(tables):
                for rn, rows in enumerate(cells):
                    for cn, c in enumerate(rows):
                        c = (c[0]+int(bbox[0]),c[1]+int(bbox[1]),c[2]+int(bbox[0]),c[3]+int(bbox[1]))
                        self.show_img = cv2.rectangle(self.show_img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 0), 3)
                        self.show_img = cv2.putText(self.show_img, '{}: c={} r={}'.format(tn, cn, rn), (int(c[0]), int(c[1])),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                           thickness=2, lineType=cv2.LINE_AA)
        else:
            print('Skip Draw {}/{}'.format(name,self.draw))
        if len(tables)>0:
            cells = tables[0]
            for i,r in enumerate(cells):
                for j,c in enumerate(r):
                    b = cells[i][j]
                    cells[i][j] = (b[0]+int(bbox[0]),b[1]+int(bbox[1]),b[2]+int(bbox[0]),b[3]+int(bbox[1]))
            return cells
        return None
    def table_second(self,coi, cells):
        if len(cells) < 4:
            return coi
        ncells = [c[-6:] for c in cells]
        ncells = self.subtable(self.get_bbox(ncells,0),'secod_table')
        if ncells is not None:
            print('use subtables')
            cells = ncells

        cells = cells[2:len(cells) - 1]
        if len(cells) < 5:
            return coi
        #cells = join_cells(cells)
        bb = self.get_bbox(cells, 0)
        data = self.extact_text(bb)
        show_img = [self.show_img]
        def _get_limits(i1, i2, names):
            amounts = []
            #namesbb = get_bbox(cells[i1:i2], img, 0)
            #tname = get_text(namesbb, data)
            #print('Name: {}-{} / {}'.format(i1,i2,tname))
            for i, row in enumerate(cells[i1:i2]):
                amount_bb = self.get_bbox([[row[-1]]], 0)
                show_img[0] = cv2.rectangle(show_img[0], (int(amount_bb[0]), int(amount_bb[1])), (int(amount_bb[2]), int(amount_bb[3])), (0, 0, 255), 2)
                def _amount(x):
                    try:
                        return float(x)
                    except:
                        return 0

                amount = self.get_text(amount_bb, data, lambda c: c.isdigit(), _amount)

                name = names.get(i, '')
                name_bb = self.get_bbox([[row[-2]]], 0)
                tname = self.get_text(name_bb, data)
                #print('Name: {}-{} / {}'.format(i1,i2,tname))
                if name == '':
                    name_bb = self.get_bbox([[row[-2]]], 0)
                    name = self.get_text(name_bb, data)
                if name != '':
                    amounts.append({'name': name, 'value': amount})
            return amounts

        def _policy_date(j, i1, i2):
            bb = self.get_bbox([[c[j]] for c in cells[i1:i2]], 0)
            return self.get_text(bb, data, lambda c: (c.isdigit() or c == ':' or c == '-' or c == '/'))

        def _policy_start_date(i1, i2):
            return _policy_date(-4, i1, i2)

        def _policy_end_date(i1, i2):
            return _policy_date(-3, i1, i2)

        def _policy_number(i1, i2):
            bb = self.get_bbox([[c[-5]] for c in cells[i1:i2]], 0)
            show_img[0] = cv2.rectangle(show_img[0], (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
            return self.get_text(bb, data, lambda c: (c.isalpha() or c.isdigit()) and c != ' ')

        def _policy_data(i1, i2, names={}):
            if len(cells) < i1:
                return None
            if len(cells) < i2:
                i2 = len(cells)
            n = _policy_number(i1, i2)
            start = _policy_start_date(i1, i2)
            end = _policy_end_date(i1, i2)
            limits = _get_limits(i1, i2, names)
            return {
                'policy': n,
                'start': start,
                'end': end,
                'limits': limits,
            }

        general = {0: 'EACH OCCURRENCE', 1: 'DAMAGE TO RENTED PREMISES (Ea Occurrence)', 2: 'MED EXP (Anyoneperson)',
                   3: 'PERSONAL & ADV INJURY', 4: 'GENERAL AGGREGATE', 5: 'PRODUCTS - COMP/OP AGG'}
        auto = {0: 'COMBINED SINGLE LIMIT', 1: 'BODILY INJURY (Per person)', 2: 'BODILY INJURY (Per accident)',
                3: 'PROPERTY DAMAGE (Per accident)'}
        umbrela = {0: 'EACH OCCURRENCE', 1: 'AGGREGATE'}
        worker = {0: 'STATUTE OTHER', 1: 'E.L. EACH ACCIDENT', 2: 'E.L. DISEASE - EA EMPLOYEE',
                  3: 'E.L. DISEASE - POLICY LIMIT'}
        coi.Liability.append({'name': 'Commercial General Liability', 'data': _policy_data(0, 7, general)})
        coi.Liability.append({'name': 'Automobile Liability', 'data': _policy_data(7, 12, auto)})
        coi.Liability.append({'name': 'Umbrela Liability', 'data': _policy_data(12, 15, umbrela)})
        coi.Liability.append({'name': 'Worker Compensation', 'data': _policy_data(15, 19, worker)})
        return coi


    def get_text(self,bb, data, char_filter=None, text_map=None):
        text = []
        for e in data:
            b = e['bbox']
            if self.is_in(bb, b):
                t = e['text']
                if char_filter is not None:
                    tmp = []
                    for c in t:
                        if char_filter(c):
                            tmp.append(c)
                    t = ''.join(tmp)
                text.append((b, t))

        text = sorted(text, key=lambda x: (x[0][3]))
        text = self.join_lines(text)
        if text_map is not None:
            text = text_map(text)
        return text


    def join_lines(self,tb):
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


    def extact_text(self, bbox):
        to_process = self.parse_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        data = pytesseract.image_to_data(Image.fromarray(to_process), config='', output_type=pytesseract.Output.DICT)
        entry = []
        conf = data['conf']
        for i, text in enumerate(data['text']):
            if int(conf[i]) < 1:
                continue
            if text is None:
                continue
            skip = True
            for c in text:
                if c.isalnum():
                    skip = False
            if skip:
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


    def is_in(self,boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / boxBArea > 0.5


    def bb_intersection_over_union(self,boxA, boxB):
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


    def get_bbox(self,cells, skip,offset=0):
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
        minx = max(0,minx-offset)
        miny = max(0,miny-offset)
        maxx = min(self.original_img.shape[1],maxx+offset)
        maxy = min(self.original_img.shape[0],maxy+offset)
        return (minx, miny, maxx, maxy)
