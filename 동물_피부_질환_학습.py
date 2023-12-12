# -*- coding: utf-8 -*-
"""동물 피부 질환 학습.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sGXSB7AwLNOh7fe7mcPH1NlakP-K7uJo
"""


"""드라이브 내 데이터 다운받기"""

#!gdown https://docs.google.com/uc?export=download&id=1KOkWMPHNrUHMth3aRK0SP9fl_WSCfF0d

#세그먼트 학습용 전처리 데이터
#!gdown https://docs.google.com/uc?export=download&id=1k7vw672QijpuTuG2NN6g7OWF8_6E4_bg


import zipfile
import os
from tqdm import tqdm
import random
import shutil
import json
from PIL import Image
import yaml
import torch
import torchvision

# gpu 연결 확인 및 파이토치 gpu버전 확인
print(torch.cuda.is_available())
print(torch.__version__)
print(torchvision.__version__)

"""### 파일 압축해제"""

# 드라이브 마운트의 경우 압축 해제
zip_file_path = '/content/drive/MyDrive/Test/2_라벨링데이터_231024_add/VL01.zip'  # 압축 파일의 경로를 적절히 설정해주세요.
# 드라이브 데이터를 다운받았을 경우
# zip_file_path = '/content/VL01.zip'

# 압축을 해제할 디렉토리 경로
extracted_dir_path = '/content/라벨링데이터'  # 압축을 해제할 디렉토리 경로를 적절히 설정해주세요.

# Zip 파일 압축 해제 with tqdm
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # 파일 수를 tqdm으로 전달하여 진행도를 표시
    file_count = len(zip_ref.namelist())
    for file in tqdm(zip_ref.infolist(), desc='압축 해제 진행 중', total=file_count):
        zip_ref.extract(file, extracted_dir_path)

# 압축 해제된 파일 목록 확인
extracted_files = os.listdir(extracted_dir_path)
print("압축 해제된 파일 목록:", extracted_files)


"""## 모델 학습"""
# yaml 파일 생성
data = { 'train': r'C:\pyprj\SET_data\train',
        'val': r'C:\pyprj\SET_data\val',
         'test': r'C:\pyprj\SET_data\test',
         'names': ['A1_구진_플라크', 'A2_비듬_각질_상피성잔고리', 'A3_태선화_과다색소침착','A4_농포_여드름','A5_미란_궤양','A6_결절_종괴','A7_무증상'],
         'nc':7
         }
with open('C:/pyprj/custom.yaml','w')as f:
  yaml.dump(data, f)

with open('C:/pyprj/custom.yaml','r')as f:
  Custom_yaml = yaml.safe_load(f)
  print(Custom_yaml)

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')

print(type(model.names), len(model.names))
print(model.names)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model.train(data = r'C:\pyprj\custom.yaml', epochs=160, patience=50, batch=256, imgsz=640)

print(type(model.names), len(model.names))
print(model.names)

# Commented out IPython magic to ensure Python compatibility.
# TensorBoard 실행
# %load_ext tensorboard
# %tensorboard --logdir runs/detect/train/

results = model.predict(source =r'C:\pyprj\SET_data\test', save=True)