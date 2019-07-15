import os
import glob
from accord.pdf import Parser
import cv2
import json

def do_parse():
    if not os.path.exists('./result_xml'):
        os.mkdir('./result_xml')

    for f in glob.glob('./images/*.PDF'):
        name = os.path.basename(f)
        print('{}'.format(name))
        p = Parser(doc=f,draw=['secod_table'])
        coi,img = p.parse()
        entries = []
        for e  in p.entries:
            entries.append({'bbox':e[0],'text':e[1]})
        with open('./result_xml/{}.entries.json'.format(name),"w") as jf:
            json.dump(entries,jf,indent=2)
        with open('./result_xml/{}.entries.xml'.format(name),"w") as jf:
            jf.write(p.xml)
        print(coi.__dict__)
        with open('./result_xml/{}.json'.format(name),"w") as jf:
            json.dump(coi.__dict__,jf,indent=2)
        cv2.imwrite('./result_xml/{}.jpg'.format(name), img)


if __name__ == '__main__':
    do_parse()
