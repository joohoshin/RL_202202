import gym
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import A2C   # A2C 모델을 불러옴

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # kernel 오류 발생 해서 넣은 것임

env = gym.make('CartPole-v1')  # Cartpole 환경을 불러옴

### 모델 학습
model = A2C('MlpPolicy', env, verbose=1) #A2C 모델 불러오기, 환경 지정
model.learn(total_timesteps=1000)  # 불러온 모델을 학습함

### 학습한 모델을 바탕으로 화면출력해보기
obs = env.reset()  # 모델 초기화
res = []  # 결과값 저장용
rewards = 0   # 리워드 확인용

for i in range(1000):  # 에피소드 반복
    action, _state = model.predict(obs)  # Obervation에 대해 모델에서 Action 결정
    obs, reward, done, info = env.step(action)
    res.append({'new_obs':obs, 'rewards':rewards})
    rewards += reward
    env.render()
    if done:  # 기준을 넘어서면 환경 초기화하고 다시 시작
        rewards = 0
        obs = env.reset()
env.close()

df = pd.DataFrame(res)
plt.plot(df.rewards)  # 최대 횟수를 확인 
model.policy

