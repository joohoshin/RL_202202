'''
모델을 살펴봅시다
'''

import gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


'''
모델 불러오기
'''
model = DQN.load("cartpole_dqn")
print(dir(model))

'''
Policy Neural Network를 살펴봅시다
'''
# torch 기본 서머리
print(model.policy)  

# tensorflow 스타일 서머리
from torchsummary import summary  
summary(model.policy, (4,)) # 모델과 인풋형태 넣어주기

'''
Observation 4개가 인풋으로 들어감
최종 선택 2가지 Action 가능하도록 2개의 아웃풋
중간에는 일반적인 Hidden Layer
'''

'''
파라미터를 가져옵시다
'''
params = model.get_parameters()
print(params['policy'])
print(params['policy.optimizer'])


