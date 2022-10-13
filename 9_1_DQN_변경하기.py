
import gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

''' 
dqn 모델의 policy network를 변화시켜 성과를 비교해봅시다
'''

env = gym.make('CartPole-v1')
model = DQN('MlpPolicy', env, verbose=1) # DQN 모델 불러오기
model.learn(total_timesteps=100000)  

   
''' 
dqn 모델을 평가해봅시다
https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
'''
mean_reward, std_reward = evaluate_policy(model, model.get_env(),
                                          n_eval_episodes=10)
print(mean_reward)

# 현재 모델 구조 확인
from torchsummary import summary  
summary(model.policy, (4,)) 

'''
policy network를 변경해봅시다
Hidden Layer를 [64,64] --> [32,32,32]로 변경해보겠습니다.
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
'''

env = gym.make('CartPole-v1')

# policy network 구조 변경
policy_kwargs = {'net_arch':[32,32,32]}
# policy_kwargs = dict(net_arch=[32,32,32]) # 위와 동일함
model = DQN('MlpPolicy', env, verbose=1, policy_kwargs = policy_kwargs) # DQN 모델 불러오기
model.learn(total_timesteps=100000)  

from torchsummary import summary  
summary(model.policy, (4,)) 

mean_reward, std_reward = evaluate_policy(model, model.get_env(),
                                          n_eval_episodes=10)
print(mean_reward)

# 다른 수치도 변경해봅시다. 
# ?DQN 출력하여 DQN의 입력값의 기본값을 확인합니다. 
# gamma와 target_update_interval을  변경해보겠습니다. 

model = DQN('MlpPolicy', env, verbose=1, policy_kwargs = policy_kwargs,
            gamma = 0.9, target_update_interval=1000)
model.learn(total_timesteps=100000)  


mean_reward, std_reward = evaluate_policy(model, model.get_env(),
                                          n_eval_episodes=10)
print(mean_reward)





