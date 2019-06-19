import voc.utils as utils
import glob
import os
import cv2

if __name__ == '__main__':
    if not os.path.exists('./test'):
        os.mkdir('./test')
    for f in glob.glob('./data/*.xml'):
        name = os.path.basename(f)
        name = name.replace('.xml','.png')
        img = cv2.imread(os.path.join('./data',name), cv2.IMREAD_COLOR)
        boxes = utils.generate(f)
        #img = utils.draw_mask(img,boxes,['Limits2Column'])
        #cv2.imwrite(os.path.join('./test',name),img)
        img = utils.gen_mask((img.shape[0],img.shape[1]),boxes,['Limits2Column'])
        img = utils.rotateImage(img,3)
        cv2.imwrite(os.path.join('./test',name),img[:,:,8])