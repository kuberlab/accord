import utils.lattice as lattice
import glob
import os
import cv2
import accord.parse as parse
import pdf2image
import numpy as np
import ocr.gunin as gunin
from tesseract.ocr import TesseractParser
from aws.ocr import AWSParser

def do_parse():
    if not os.path.exists('./result_tables'):
        os.mkdir('./result_tables')
    number_fn = gunin.get_number_fn('./ocr-numbers')
    #number_fn = None
    ocr = TesseractParser()
    #ocr = AWSParser()
    for f in glob.glob('./images/*'):
        name = os.path.basename(f)
        print('{}'.format(name))
        img = cv2.imread(os.path.join('./images', name), cv2.IMREAD_COLOR)
        if img is None:
            pages = pdf2image.convert_from_path(os.path.join('./images', name), 300)
            name = name.replace('.pdf','')
            for i,p in enumerate(pages):
                print("\n\nprocess - {}".format(i+1))
                img  = np.array(p,np.uint8)
                img = img[:,:,::-1]
                p = parse.Parser(img,ocr,draw=['first_table'],number_ocr=number_fn)
                coi,img = p.parse()
                cv2.imwrite(os.path.join('./result_tables', '{}-{}.jpg'.format(name,i+1)), img)
                print(coi.__dict__)
            continue
        p = parse.Parser(img,ocr,draw=['first_table'],number_ocr=number_fn)
        coi,img = p.parse()
        cv2.imwrite(os.path.join('./result_tables', name), img)
        print(coi.__dict__)

def test():
    if not os.path.exists('./result_tables'):
        os.mkdir('./result_tables')
    for f in glob.glob('./images/*'):
        name = os.path.basename(f)
        print('{}'.format(name))
        img = cv2.imread(os.path.join('./images', name), cv2.IMREAD_COLOR)
        print('{}: {}'.format(name, img.shape))
        tables, segments = lattice.extract_tables(img)
        for tn, cells in enumerate(tables):
            rn = 0
            for rn, rows in enumerate(cells):
                for cn, c in enumerate(rows):
                    img = cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 0), 2)
                    img = cv2.putText(img, '{}: c={} r={}'.format(tn, cn, rn), (int(c[0]), int(c[1])),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                      thickness=2, lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join('./result_tables', name), img)


if __name__ == '__main__':
    do_parse()