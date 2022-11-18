<h1 align="center">
  <img src="labelme/icons/icon.png"><br/>Automated labelme
</h1>

<h4 align="center">
  Image Annotation Tool with Automated <a href="https://github.com/wkentaro/labelme">labelme<a/> in Python 
</h4>

<div align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/torch.svg"></a>
</div>
<div align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"></a>
</div>

<div align="center">
  <a href="#installation"><b>Installation</b></a> |
  <a href="#usage"><b>Usage</b></a> |
  <a href="https://github.com/wkentaro/labelme/tree/main/examples/tutorial#tutorial-single-image-example"><b>Tutorial</b></a> |
  <a href="https://github.com/wkentaro/labelme/tree/main/examples"><b>Examples</b></a> |
  <a href="https://github.com/wkentaro/labelme/discussions"><b>Discussions</b></a> |
  <a href="https://www.youtube.com/playlist?list=PLI6LvFw0iflh3o33YYnVIfOpaO0hc5Dzw"><b>Youtube FAQ</b></a>
</div>

<h1 align="center">
  <img src="/images/automated-labelme.jpg" width="75%"><br/>
</h1>

<br/>

## Description

Labelme is a graphical image annotation tool inspired by <http://labelme.csail.mit.edu>. It is written in Python and uses Qt for its graphical interface. You can find this tool at: [LabelMe in Python](https://github.com/wkentaro/labelme) In addition to this tool, a plugin has been developed that will provide automatic annotation for PCM time series, which have very few annotations and are very difficult to label due to artefacts. The method available at [ConvNeXt MaskR-CNN](https://github.com/mberkay0/ConvNeXt-MaskRCNN) uses the pre-trained model for the PCM data to obtain semi-automated annotated images. As a result, you can speed up annotation with the plugin developed for PCM images. 

Also, if you want to train the model for your specific topic, see [ConvNeXt MaskR-CNN](https://github.com/mberkay0/ConvNeXt-MaskRCNN). After you prepare your model, be sure to manipulate the [detector.py](https://github.com/mberkay0/automated-labelme/blob/main/labelme/detector.py) file and the [default_config.yaml](https://github.com/mberkay0/automated-labelme/blob/main/labelme/config/default_config.yaml) file based on your work. Comments that can manage these files are available inside the files.

<div align="center">
  <img src="https://user-images.githubusercontent.com/4310419/47907116-85667800-de82-11e8-83d0-b9f4eb33268f.gif" width="30%" /> <img src="https://user-images.githubusercontent.com/4310419/47922172-57972880-deae-11e8-84f8-e4324a7c856a.gif" width="30%" /> <img src="https://user-images.githubusercontent.com/14256482/46932075-92145f00-d080-11e8-8d09-2162070ae57c.png" width="32%" />  
  
<i>Various primitives (polygon, rectangle, circle, line, and point).</i>
</div>

## Features

- [x] Image annotation for polygon, rectangle, circle, line and point. 
- [x] Image flag annotation for classification and cleaning. ([#166](https://github.com/wkentaro/labelme/pull/166))
- [x] Video annotation. 
- [x] GUI customization (predefined labels / flags, auto-saving, label validation, etc). ([#144](https://github.com/wkentaro/labelme/pull/144))
- [x] Exporting VOC-format dataset for semantic/instance segmentation.
- [x] Exporting COCO-format dataset for instance segmentation. 
- [x] Pretrained Model for semi-auto image annotation [Drive link for PCM Cell Detection (bbox and segmentation auto annotation)](https://drive.google.com/file/d/1I5NXHJXqMYjLBf6HSCDWvYAnZx_X8b8v/view?usp=share_link)
- [x] Ease of use in automatic detection.
- [x] Useful for fast data labelling in data-scarce environments such as cell detection, segmentation and tracking.



## Requirements

- Linux / macOS / Windows
- Python3.7 or more
- PyQt5 / PySide2
- Pytorch


## Installation


### Linux, Windows and MacOS

```bash
git clone https://github.com/mberkay0/automated-labelme.git
  
cd automated-labelme
  
python -m venv env
# then activate virtual env.
#.\env\Scripts\activate #Windows
#./env/bin/activate #Linux, MacOS

pip install .
# then use from command line 
labelme
```


## Usage

Before you begin, copy the pre-trained model into the [model folder](https://github.com/mberkay0/automated-labelme/tree/main/model). If you want to use the model used in the project, download it via the [link](https://drive.google.com/file/d/1I5NXHJXqMYjLBf6HSCDWvYAnZx_X8b8v/view?usp=share_link).
  
* You can select the detection type from the "Select detection type" section under the "Detection" menu, either 'bbox' or 'mask'. By default, it comes mask.

<div align="center">
  <img src="/images/labelme_detection_options.jpg" width="75%"><br/>
</div>
  
* Then use the buttons below to load the data.
 
<div align="center">
  <img src="/images/open_data.jpg" width="35%"><br/> 
</div>

<div align="center">
  <img src="/images/semi_labelme_cell_annotation1.jpg" width="75%"><br/>
</div>

* Next, let's look at other settings for automatic annotation.

<div align="center">
  <img src="/images/detection_options.jpg" width="60%"><br/>
</div>

* If you have selected the detection type mask, you can change the number of samples. This value is how often points to be detected are annotated. It is between 0 and 120. Deals close to 0 allow more points to be seen. Deals close to 120 enable fewer points to be seen. If 0 is given, all detected points are created. By default, a value of 25 comes. Enter values close to 0 to make a more detailed annotation.

<div align="center">
  <img src="/images/subsample_option.jpg" width="55%"><br/>
</div>

* Enter a value between 0 and 1 to access reliable results. A value close to 1 is the result that the model predicted with higher confidence. Enter a value between 0 and 1 to threshold the probability map generated by the model.
  
<div align="center">
  <img src="/images/threshold_option.jpg" width="55%"><br/>
</div>

* Press the Detect button for automatic annotation and wait a bit. All automatically caught detections will be annotated in the specified type.
 
<div align="center">
  <img src="/images/semi_labelme_cell_annotation2.jpg" width="75%"><br/>
</div>

* Finally, you can get the automatic detection results below.
 
<div align="center">
  <img src="/images/semi_labelme_cell_annotation3.jpg" width="50%"><img src="/images/bbox.jpg" width="50%">
  
<i>Automated annotation result types (polygon, rectangle).</i>
</div>
  

Run `labelme --help` for detail.
The annotations are saved as a [JSON](http://www.json.org/) file like COCO dataset.


## How to build standalone executable

Below shows how to build the standalone executable on macOS, Linux and Windows.  

```bash
# clone the repo
git clone https://github.com/mberkay0/automated-labelme.git
  
cd automated-labelme
  
python -m venv env
# then activate virtual env.
#.\env\Scripts\activate #Windows
#./env/bin/activate #Linux, MacOS

# Build the standalone executable
pip install .
pip install pyinstaller
pyinstaller labelme.spec
```


### Command Line Arguments
- `--output` specifies the location that annotations will be written to. If the location ends with .json, a single annotation will be written to this file. Only one image can be annotated if a location is specified with .json. If the location does not end with .json, the program will assume it is a directory. Annotations will be stored in this directory with a name that corresponds to the image that the annotation was made on.
- The first time you run labelme, it will create a config file in `~/.labelmerc`. You can edit this file and the changes will be applied the next time that you launch labelme. If you would prefer to use a config file from another location, you can specify this file with the `--config` flag.
- Without the `--nosortlabels` flag, the program will list labels in alphabetical order. When the program is run with this flag, it will display labels in the order that they are provided.

For more details, check the forked repo.



## Acknowledgement

This repo is the fork of [wkentaro/pylabelme](https://github.com/wkentaro/labelme).
  
* http://labelme.csail.mit.edu

* [LabelMe The Open annotation tool](https://github.com/CSAILVision/LabelMeAnnotationTool)

* [mpitid/pylabelme](https://github.com/mpitid/pylabelme)

* [labelme v5.1.0](https://github.com/wkentaro/labelme)

* [torchvision](https://github.com/pytorch/vision)

* [ConvNeXt MaskR-CNN](https://github.com/mberkay0/ConvNeXt-MaskRCNN)
