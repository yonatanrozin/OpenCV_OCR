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
  - Be sure to include path to tesseract.exe in system PATH. Test installation by running 'tesseract' in prompt
- ```git clone``` this repo into desired location
 
## Usage
### OpenCV camera setup (if using with webcam)
Code to open camera will differ per device. Change your camera index by changing the line ```cam_index = 0``` towards the top of OCR.py. Run ```OCR.py list``` to get a list of available camera indexes. Additionally, your camera may require a specific video capture API. You can use one by uncommenting one of the ```capture_api = ...``` lines below the cam_index line. More info about capture APIs can be found in the [OpenCV capture API docs](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d).
### Running
- ```cd``` into cloned directory
- run ```python3 OCR.py <arg> <filepath (if needed)> <debug (optional)>``` in a command prompt
  - ```<arg>```: can be one of ```list```, ```image```, ```video``` or ```webcam```
    - List: list available camera indexes
    - Image/video/webcam: specify media to scan for text. If using image or video, the following argument must be the path to the file. __Tested with .jpg and .mov photo/video formats. Other formats have not yet been tested.__
  - ```<filepath>```: used to specify a path to the image/video file to be scanned. You can drag and drop the file into the command prompt window to get the file path automatically.
  - ```<debug>```: include ```debug``` as the last argument to run in debug mode.
    - Opens additional windows containing intermediate OpenCV matrices generated during text scanning
    - Prints additional status messages to console, as well as writing them to a ```log.txt``` file.
   
## Pytesseract troubleshooting
Tesseract must be installed and added to system PATH before pytesseract will work. See [here](https://tesseract-ocr.github.io/tessdoc/Installation.html) for installation instructions. The installation can be tested by running ```tesseract``` in a command prompt. If tesseract has been installed but doesn't work in Python, uncomment the line ```pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>``` in OCR.py, providing the absolute path to the ```tesseract.exe``` file in the Tesseract directory.
