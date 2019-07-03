import json
from trp import Document

def processDocument(doc):
    for i,page in enumerate(doc.pages,1):
        print("PAGE: {}\n====================".format(i))
        for line in page.lines:
            #print("Line: {}--{}".format(line.text, line.confidence))
            for word in line.words:
                #print("Word: {}--{}".format(word.text, word.confidence))
                None


def run():
    response = {}

    filePath = "aws1.json"
    with open(filePath, 'r') as document:
        response = json.loads(document.read())

    doc = Document(response)
    processDocument(doc)

run()
