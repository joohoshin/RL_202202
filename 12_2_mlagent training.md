### Unity mlagent 학습하기

https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Getting-Started.md

* ppo 설정 파일 확인
  - git clone하여 복사한 폴더 안에 config 폴더에 설치 파일이 yaml로 저장되어 있음
  - 3dballs.yaml 파일에서 설정 변경하여 ppo로 훈련 가능함
  - yaml 파일 편집은 VS Code에서 하면 편함

* 학습하기
  - mlagents-learn ml-agents/config/ppo/3DBall.yaml --run-id=first3DBallRun
  - yaml 파일 있는 경로 정확히 입력 필요
  - --run-id는 이름 지정해두면 추후 추가 학습 가능함
  - mlagents-learn ml-agents/config/ppo/3DBall.yaml --run-id=first3DBallRun --resume
    : --resume을 추가하면 id에 맞게 학습이 이어서 진행 됨
  - 실행되면 unity 글자와 함께 play버튼 누르라는 안내가 나옴
  - unity에서 학습이 진행되는 모습을 볼 수 있다 

* 결과 확인하기
  - 학습 후에 onnx 파일이 생성됨
  - onnx는 tensorflow, pytorch 등 다양한 프레임워크의 모델의 통합 표준
  - 학습된 모델을 TFModels 폴더에 복사해서 사용 가능함
  - prefab에서 학습된 모델을 변경하여 사용 가능함

* tensorboard 확인
  - tensorboard --logdir results 
  - 결과 폴더는 실행한 폴더 아래 results로 저장됨
