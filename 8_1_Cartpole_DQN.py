
import gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

''' 
dqn 모델을 학습해봅시다
'''

# RL 모델 선택
env = gym.make('CartPole-v1')
model = DQN('MlpPolicy', env, verbose=1) # DQN 모델 불러오기
model.learn(total_timesteps=100000)  
   # total_timesteps가 적으면 잘 학습이 되지 않습니다. 
   
''' 
dqn 모델을 평가해봅시다
https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
'''
mean_reward, std_reward = evaluate_policy(model, model.get_env(),
                                          n_eval_episodes=10)

'''
모델 저장하기 
'''
model.save("cartpole_dqn")
''' 
학습된 모델로 실행해서 화면으로 확인해봅시다. 
'''

# 아래 오류 발생 시에 아래 2줄 필요
# OMP: Error #15: Initializing libiomp5md.dll...

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


env = gym.make('CartPole-v1') # 화면 출력
# env = gym.make('CartPole-v1', render_mode = 'human') # 버전에 따라 화면 출력 시 필요 
# pip show gym   패키지 버전 확인, prompt에서 실
obs = env.reset()
res = []
rewards = 0   
for i in range(1000):  # 1000 time step 반복
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    res.append({'new_obs':obs, 'rewards':rewards})
    rewards += reward
    env.render()
    if done:
        rewards = 0
        obs = env.reset()
env.close()

df = pd.DataFrame(res)
plt.plot(df.rewards)  # 최대 횟수를 확인 

''' 
랜덤하게 선택하는 것과 결과를 비교해봅시다. 
'''
env = gym.make('CartPole-v1', render_mode='human') # 화면 출력
obs = env.reset()
res = []
rewards = 0   
for i in range(1000):
    action = env.action_space.sample()  # 랜덤하게 선택하기
    obs, reward, done, info = env.step(action)
    res.append({'new_obs':obs, 'rewards':rewards})
    rewards += reward
    env.render()
    if done:
        rewards = 0
        obs = env.reset()
env.close()

df = pd.DataFrame(res)
plt.plot(df.rewards)  # 최대 횟수를 확인 





