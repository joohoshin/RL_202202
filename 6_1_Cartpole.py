
'''
Hello World
https://www.gymlibrary.ml/environments/classic_control/cart_pole/

env 주요 기능
reset: This function resets the environment to its initial state, and returns the observation of the environment corresponding to the initial state.
step : This function takes an action as an input and applies it to the environment, which leads to the environment transitioning to a new state. The reset function returns four things:
observation: The observation of the state of the environment.
reward: The reward that you can get from the environment after executing the action that was given as the input to the step function.
done: Whether the episode has been terminated. If true, you may need to end the simulation or reset the environment to restart the episode.
info: This provides additional information depending on the environment, such as number of lives left, or general information that may be conducive in debugging.
'''

import gym

env = gym.make('CartPole-v1')
env.reset()
env.render()


# 환경을 살펴봅시다. 
# print(dir(env))
# print(env.action_space)  
# print(env.observation_space)

'''
action_space는 Discrete(2):0,1
observation_space Box로 범위가 지정되어 있음
   4개의 값에 대해서 최소 범위와 최대 범위가 있음
'''


# 랜덤으로 액션을 지정해봅시다.     
action= env.action_space.sample() # 0,1에서 선택
    
# step을 진행하여 결과를 받아봅시다
new_obs, reward, done, info = env.step(action)
print(new_obs)
print(reward)
print(done)
print(info)

env.render()


env.close()


'''
여러번 반복해서 움직이게 해봅시다. 
이때 살아있으면 reward는 1을 반환합니다
 
'''

import time
env = gym.make('CartPole-v1')
env.reset()   #초기화
env.render()
res = []
rewards = 0
for i in range(1000):
    action = env.action_space.sample() # 0,1에서 선택
    new_obs, reward, done, info = env.step(action)
    rewards += reward
    res.append({'new_obs':new_obs, 'rewards':rewards})
    #time.sleep(0.01)
    env.render()
    if done:
        rewards = 0
        env.reset()
        
env.close()

'''
pandas dataframe으로 변경하여 살펴봅시다
'''

import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(res)
plt.plot(df['rewards']) # 최대 횟수를 확


