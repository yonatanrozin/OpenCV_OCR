# OpenCV_OCR
Performs OCR from webcam feed, image or video file.

## Dependencies and installation
- [Python](https://www.python.org/) - tested with version 3.11.6
- Python packages (use ```pip``` to install)
  - [Scipy](https://pypi.org/project/scipy/) - tested with version 1.11.3
  - [Numpy](https://pypi.org/project/numpy/) - tested with version 1.24.3
  - [OpenCV-Python](https://pypi.org/project/opencv-python/) - tested with version 4.8.1
  - [pytesseract](https://pypi.org/project/pytesseract/) - tested with version 0.3.10
  - __Be sure [OpenCV-Python-Headless](https://pypi.org/project/opencv-python-headless/) is not installed!__ 
- Tesseract OCR - tested with version 5.3.3
  - See installation instructions [here](https://tesseract-ocr.github.io/tessdoc/Installation.html)
  - Be sure to include path to tesseract.exe in system PATH. Test installation by running 'tesseract' in a command prompt
- ```git clone``` this repo into desired location
 
## Usage
- ```cd``` into cloned directory
- run ```python3 OCR.py <arg> <filepath (if needed)> <debug (optional)> in a command prompt
  - ```<arg>```: can be one of ```list```, ```image```, ```video``` or ```webcam```
    - List: list available camera indexes
    - Image/video/webcam: specify media to scan for text. If using image or video, the following argument must be the path to the file.
  - ```<filepath>```: used to specify a path to the image/video file to be scanned. You can drag and drop the file into the command prompt window to get the file path automatically.
  - ```<debug>```: include ```debug``` as the last argument to run in debug mode.
    - Opens additional windows containing intermediate OpenCV matrices generated during text scanning
    - Prints additional status messages to console, as well as writing them to a ```log.txt``` file.
