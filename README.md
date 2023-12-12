# pet-skin-disease_yolov8
## 프로젝트 소개
반려동물 피부 질환데이터를 가지고 피부질환 웹 서비스를 구축하기 위해 필요한 AI모델 구축
![스크린샷 2023-11-29 165434](https://github.com/Leejujin/pet-skin-disease_yolov8/assets/38245868/d52ef93f-a72e-477a-aef3-63e16e0831e3)
### 전처리된 데이터 다운로드 링크(실행 코드에도 다운로드 url 있습니다.) => 피부 질환 데이터
[전처리 데이터 구글 드라이브 다운로드 링크](https://docs.google.com/uc?export=download&id=1KOkWMPHNrUHMth3aRK0SP9fl_WSCfF0d)
### 관련 패키지 설치
(cuda버전 필수 확인 , 해당 버전과 맞는 torch,torchvision 버전 설치)
```python
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm
pip install Pillow
pip install PyYAML
pip install ultralytics  #yolov8 관련 패키지
pip install gdown  # 대용량 파일 다운로드 관련 패키지
```
## yolov8
![image](https://github.com/Leejujin/pet-skin-disease_yolov8/assets/38245868/b40fa40b-8653-423e-bf93-282765ee4510)
YOLO의 최신 버전인 YOLOv8는 이전 버전의 성공을 기반으로 새로운 기능과 개선 사항을 도입하여 성능과 유연성을 더욱 향상시켰다.
YOLOv8은 도큐먼트화 및 모듈화가 친화적으로 잘 되어 있어 사용에 용이하다.

### 커스텀 데이터
원천데이터 : AI hub [반려동물 피부 질환 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=561)

데이터의 전처리는 코랩 환경에서 데이터 처리 했습니다.

이미지 파일의 개수는 다음과 같습니다.

![스크린샷 2023-11-29 154009](https://github.com/Leejujin/pet-skin-disease_yolov8/assets/38245868/c9558ff9-7a0d-42c0-a318-98e4ad6b2f8d)

A7_무증상 같은 경우 각각의 질병의 무증상 이미지이므로 각각의 질병 무증상 500장씩 총3000장을 데이터로 사용했습니다.

### 코드 설명
  * 압축 해제
    - 다운로드 받은 파일을 압축해제하는 코드가 들어있습니다. 압축을 해제할 경로를 해당 환경에 맞게 설정해주세요
    - (7-zip 같은 프로그램으로 푸셔도 됩니다. 그러할 경우 해당 코드는 삭제하시고 거나 shift+alt+e로 특정 셀만 실행해주세요)
  
  * yaml파일 생성
    - 경로들은 해당 환경에 맞게 수정해야 합니다.(혹시 몰라 yaml파일 첨부하였습니다)
    - ```python
      # yaml 파일 생성
      data = {'train': r'C:\pyprj\SET_data\train',
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
      ```
  * yolov8 모델 로드
    - 사전 학습된 모델 yolov8n.pt 을 로드합니다.(yolov8s.pt도 사용가능 해당 환경이 좀더 무거운 모델을 써도 된다면 사용권장)
    - ```python
      from ultralytics import YOLO
      model = YOLO('yolov8n.pt')
      ```
  * 커스텀 데이터 학습
    - yaml파일의 경로는 해당 환경에 맞게 수정해야합니다. (버전이 다른 s,m,l,x 같은 경우 model = 'yolov8s.pt' 와 같이 지정해 주어야 한다.)
    - ```python
      model.train(data = r'C:\pyprj\custom.yaml', epochs=150, patience=50, batch=32, imgsz=416)
      ```
    - // eopchs => 훈련 시킬 횟수
    - // patience => 관촬될만한 Training 진전이 없으면 30번 보다가 Training 중단
    - // batch => 일괄입력 data 개수
    - // imgsz => input image 크기
    - // optimizer => 옵티마이저 설정 (SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto)

  * 테스트 이미지 예측
    - test 경로는 해당 컴퓨터 환경에 맞게 수정해야합니다.(압축을 푸셨던 디렉토리 경로로 설정해주세요)
    - ```python
      results = model.predict(source =r'C:\pyprj\SET_data\test', save=True)
      ```
#### 예측되어 나오는 원하는 결과물
![IMG_D_A2_603230](https://github.com/Leejujin/pet-skin-disease_yolov8/assets/38245868/a865791a-2f82-40cb-a849-4556f505c829)
