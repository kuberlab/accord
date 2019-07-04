import pytesseract
import PIL.Image as Image

class TesseractParser(object):
    def __init__(self):
        None
    def prepare_image(self,parse_img):
        parse_img[parse_img<150] = 0
        return parse_img
    def extact_text(self,parse_img,bbox,offset=0,th=-1):
        to_process = parse_img[bbox[1]-offset:bbox[3]+offset, bbox[0]-offset:bbox[2]+offset, :]
        data = pytesseract.image_to_data(Image.fromarray(to_process), config='', output_type=pytesseract.Output.DICT)
        entry = []
        conf = data['conf']
        for i, text in enumerate(data['text']):
            if int(conf[i]) < th:
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
            left = data['left'][i] + bbox[0]-offset
            top = data['top'][i] + bbox[1]-offset
            width = data['width'][i]
            height = data['height'][i]
            entry.append({'bbox': [left, top, left + width, top + height], 'text': text,'confidence': conf[i]})
        return entry