
 
## Corona Monitor txt recognition

This repository is a simplified version of [this](https://github.com/zvikapeter/deep-text-recognition-benchmark). 

###  Environment setup
- This work was tested with PyTorch 1.4.0, CUDA 10.0, python 3.6 and Ubuntu 16.04. 
-  Clone the repository
```
git clone https://github.com/moshes7/Medical-Monitor-OCR.git
cd Medical-Monitor-OCR/algorithms/deep-text-recognition-simple
```
- Requirements 
```
pip3 install -r requirements.txt
```
### Run demo with pretrained model

1. Download pretrained model ```TPS-ResNet-BiLSTM-Attn.pth``` from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW) to the pre trained folder in ```deep-text-recognition-benchmark\PreTrained```
2. From ```deep-text-recognition-benchmark\PreTrained``` run ``` python monitor_ocr.py```
