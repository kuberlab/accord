FROM kuberlab/serving:latest

RUN add-apt-repository ppa:alex-p/tesseract-ocr
RUN apt-get update && apt-get install tesseract-ocr -y && \
  apt-get install poppler-utils -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install pytesseract

RUN pip install fuzzyset

RUN pip install pdf2image

RUN pip install imutils

RUN git clone https://github.com/dreyk/pdfminer.six.git && \
    cd pdfminer.six && \
    pip install .
