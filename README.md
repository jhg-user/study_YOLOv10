# study_YOLOv10
### <학교 현장실습(AI 개발 - YOLOv10 공부)>
최근 모델인 YOLOv10 논문 리뷰 후 '사람/사람아님'을 구분할 수 있는 모델을 만들고자
여러 크기의 YOLOv10 모델을 분석한 후, 가장 효율적인 성능(학습속도, 인식 정확도)을 보이는 크기를 선택함.

사람이 아니지만 사람으로 인식되는 문제점(이상치)을 해결하기 위해 anomaly detection에 대해 학습하고, YOLOv10에 추가하여 구현함.


1. YOLOv10에 wider face dataset 학습

   (참고) https://github.com/THU-MIG/yolov10


2. 학습 모델에 anomaly detection 구현
- Deep SVDD

  (참고) https://github.com/mperezcarrasco/PyTorch-Deep-SVDD
- VAE

  (참고) https://github.com/AntixK/PyTorch-VAE


-----


### 데이터셋

- UTKface

  https://susanqq.github.io/UTKFace/

- CelebA

  https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data

- wider face

  http://shuoyang1213.me/WIDERFACE/

- iCartoonFace

  https://github.com/luxiangju-PersonAI/iCartoonFace

- animalweb

  https://fdmaproject.wordpress.com/author/fdmaproject/
