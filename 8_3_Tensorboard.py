
import gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

''' 
tensorboard를 통해 학습과정을 살펴봅시다
'''

env = gym.make('CartPole-v1')
# tensorboard 저장 폴더 옵션을 넣어줍니다. 
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./cartpole_dqn/")
# tensorboad에 기록되는 이름을 넣어줍니다. 
model.learn(total_timesteps=10000, tb_log_name="1st")  

'''
학습이 덜 되었다면 이어서 진행해봅시다
'''
model.learn(total_timesteps=10000, tb_log_name="2nd", 
            reset_num_timesteps=False)  # 이어서 학습 가능하도록 옵션 설정
model.learn(total_timesteps=50000, tb_log_name="3nd", 
            reset_num_timesteps=False)  # 이어서 학습 가능하도록 옵션 설정
model.learn(total_timesteps=50000, tb_log_name="4th", 
            reset_num_timesteps=False)  # 이어서 학습 가능하도록 옵션 설정
