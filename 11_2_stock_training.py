import stock_env

'''
환경을 테스트해봅시
'''
env =stock_env.stock_env('005930')
obs = env.reset()
for i in range(100):
    action = env.action_space.sample()    
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()

'''
모델을 학습해봅시다
'''

import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = stock_env.stock_env('005930')
# policy_kwargs = {'net_arch':[64, {'vf':[64,64], 'pi':[16]}]}
model = PPO("MlpPolicy", env,verbose = 1, tensorboard_log="./number_ppo_mlp/") 
            # policy_kwargs = policy_kwargs)
obs = env.reset()
env.close()

from torchsummary import summary  
summary(model.policy, (60,)) 


model.learn(total_timesteps=10000, tb_log_name="1st")  
model.learn(total_timesteps=50000, tb_log_name="2nd", reset_num_timesteps=False)  
model.learn(total_timesteps=50000, tb_log_name="3rd", reset_num_timesteps=False)  
model.learn(total_timesteps=50000, tb_log_name="4th", reset_num_timesteps=False)  

'''
실제로 어떻게 예측하는지 저장해봅시다
'''
env =stock_env.stock_env('005930')
obs = env.reset()
prices=[]
actions = []
for i in range(1000):
    action = model.predict(obs, deterministic=True)    
    print(action[0])
    obs, rewards, done, info = env.step(action[0])
    prices.append(env._get_price())
    actions.append(action[0])
    env.render()
env.close()

'''
학습이 잘 되지 않음, 개선 방법 추가 검토 필요함
'''
