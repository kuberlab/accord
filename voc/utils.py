import xml.etree.ElementTree as ET
import cv2
import numpy as np

clazzes = {
    'Date': 0,
    'Producer': 1,
    'Insured': 2,
    'CertNumber': 3,
    'PolicyNumberColumn': 4,
    'PolicyEFFColumn': 5,
    'PolicyEXPColumn': 6,
    'Limits1Column': 7,
    'Limits2Column': 8,
    'TypeOFInsurenceColumn': 9,
    'Row': 10,
    'LimitsRow': 11,
    'Holder': 12,
    'Contact': 13,
    'ContactPhone': 14,
    'ContactFax': 15,
    'ContactEmail': 16,
}
def gen_mask(size,boxes,clazz=None):
    w = size[1]
    h = size[0]
    img = [np.zeros([h,w,3],np.uint8) for _ in clazzes.keys()]
    for b in boxes:
        if (clazz is None) or (b[0] in clazz):
            z = clazzes[b[0]]
            img[z] = cv2.rectangle(img[z],(int(b[1][0]*w),int(b[1][1]*h)),(int(b[1][2]*w),int(b[1][3]*h)),(255,0,0),cv2.FILLED)
    img = [z[:,:,0:1] for z in img]
    return np.concatenate(img,axis=2)
def draw_mask(img,boxes,clazz):
    w = img.shape[1]
    h = img.shape[0]
    for b in boxes:
        if b[0] in clazz:
            img = cv2.rectangle(img,(int(b[1][0]*w),int(b[1][1]*h)),(int(b[1][2]*w),int(b[1][3]*h)),(255,0,0),cv2.FILLED)
    return img

def generate(file):
    tree = ET.parse(file)
    root = tree.getroot()
    size_e = root.find('./size')
    width = int(size_e.find('./width').text)
    height = int(size_e.find('./height').text)
    boxes = []
    for object in root.findall('./object'):
        name = ''
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        for c in object.getchildren():
            if c.tag=='name':
                name = c.text
            elif c.tag=='bndbox':
                for b in c.getchildren():
                    if b.tag == 'xmin':
                        xmin = int(b.text)
                    elif b.tag == 'xmax':
                        xmax = int(b.text)
                    elif b.tag == 'ymax':
                        ymax = int(b.text)
                    elif b.tag == 'ymin':
                        ymin = int(b.text)
        if name!='' and (xmin,ymin,xmax,ymax)!=(0,0,0,0):
            xmin = min(xmax,xmin+1)
            ymin = min(ymax,ymin+1)
            xmax = max(xmin,xmax-1)
            ymax = max(ymin,ymax-1)
            boxes.append((name,[float(xmin)/width,float(ymin)/height,float(xmax)/width,float(ymax)/height]))
    return boxes



def rotateImage(image, angle,image_center=None):
    if image_center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result