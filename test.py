import utils.lattice as lattice
import glob
import os
import cv2
import accord.parse as parse

def do_parse():
    if not os.path.exists('./result_tables'):
        os.mkdir('./result_tables')
    for f in glob.glob('./images/*'):
        name = os.path.basename(f)
        print('{}'.format(name))
        img = cv2.imread(os.path.join('./images', name), cv2.IMREAD_COLOR)
        parse.parse(img)

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