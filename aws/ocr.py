import boto3

import time
import PIL.Image as Image
import io
from aws.trp import Document


class AWSParser(object):
    def __init__(self, bucket='kibernetika-strada', region_name='us-east-1', aws_access_key_id=None,
                 aws_secret_access_key=None):
        self.i = 0
        if aws_secret_access_key is None:
            self.s3 = boto3.resource('s3')
            self.textract = boto3.client('textract')
        else:
            self.s3 = boto3.resource('s3', region_name=region_name,
                                     aws_access_key_id=aws_access_key_id,
                                     aws_secret_access_key=aws_secret_access_key)
            self.textract = boto3.client('textract', region_name=region_name,
                                         aws_access_key_id=aws_access_key_id,
                                         aws_secret_access_key=aws_secret_access_key)
        self.bucket = bucket
        self.doc = []

    def isJobComplete(self, job_id):
        time.sleep(5)

        response = self.textract.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

        while (status == "IN_PROGRESS"):
            time.sleep(5)
            response = self.textract.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]
            print("Job status: {}".format(status))

        return status

    def getJobResults(self, job_id):

        pages = []

        time.sleep(5)

        client = boto3.client('textract')
        response = client.get_document_text_detection(JobId=job_id)

        pages.append(response)
        print("Resultset page recieved: {}".format(len(pages)))
        nextToken = None
        if ('NextToken' in response):
            nextToken = response['NextToken']

        while (nextToken):
            time.sleep(5)

            response = client.get_document_text_detection(JobId=job_id, NextToken=nextToken)

            pages.append(response)
            # break
            print("Resultset page recieved: {}".format(len(pages)))
            nextToken = None
            if ('NextToken' in response):
                nextToken = response['NextToken']

        return pages

    def _get_result(self,name,width,height):
        response = self.textract.start_document_text_detection(
            DocumentLocation={
                'S3Object': {
                    'Bucket': self.bucket,
                    'Name': name
                }
            })
        job_id = response["JobId"]
        if (self.isJobComplete(job_id)):
            response = self.getJobResults(job_id)
            doc = Document(response)
            entry = []

            for i, page in enumerate(doc.pages, 1):
                for line in page.lines:
                    entry.append({'bbox': [
                        int(width * line.geometry.boundingBox.left),
                        int(height * line.geometry.boundingBox.top),
                        int(width * (line.geometry.boundingBox.left + line.geometry.boundingBox.width)),
                        int(height * (line.geometry.boundingBox.top + line.geometry.boundingBox.height))],
                        'text': line.text,
                        'confidence': line.confidence,
                    })
            self.doc = entry
    def prepare_image(self, parse_img):
        self.i += 1
        img = Image.fromarray(parse_img)
        width, height = img.size
        arr = io.BytesIO()
        img.save(arr, 'PDF', resolution=100.0, save_all=True)
        arr = arr.getvalue()
        name = '{}-{}.pdf'.format(time.time(),self.i)
        object = self.s3.Object(self.bucket, name)
        object.put(Body=arr)
        try:
            self._get_result(name,width,height)
        except:
            self.doc = []
        object = self.s3.Object(self.bucket, name)
        object.delete()
        return parse_img

    def is_in(self, boxA, boxB, th=0.5):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if boxBArea == 0:
            return False

        return interArea / boxBArea >= th

    def extact_text(self, parse_img, bbox, offset=0, th=-1):
        doc = []
        for e in self.doc:
            b = e['bbox']
            if self.is_in(bbox, b):
                doc.append(e)
        return doc
