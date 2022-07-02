# ALPR USING VIDEO

> **NOTE:** You should download yolo_utils folder from the google drive link below and put this folder and ***ALPR.py*** file in the same folder.  

[Install yolo_utils folder for yolo model](https://drive.google.com/drive/folders/1Aea2ZMeMf9Teans9awg7AIE2bkiTWJjG?usp=sharing)
## REQUIREMENTS

- OPENCV
- PYTESSERACT
- NUMPY
- MATPLOTLIB
- ITERTOOLS


## INSTALLATION
### Installing opencv
`pip install opencv-python`

### Installing pytesseract
**1**. Install tesseract using windows installer available at: [Install tesseract for windows](https://github.com/UB-Mannheim/tesseract/wiki)
**2**. Note the tesseract path from the installation. Default installation path at the time of this edit was:  `C:\Users\USER\AppData\Local\Tesseract-OCR`. It may change so please check the installation path.

**3**.  `pip install pytesseract`

**4**. Set the tesseract path in the script before calling  `image_to_string`:

`pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'`

### Installing numpy
`pip install numpy`
### Installing matplotlib
`pip install matplotlib`
### Installing itertools
`pip install more-itertools`

## USAGE
This project uses .mp4 file, yolov3 model and pytesseract for license plate detection  

