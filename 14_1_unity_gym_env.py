'''
https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Executable.md
https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-Gym-API.md
'''

'''
unity wrapper 사용 시에는 에이전트 하나만 사용이 가능함
속도가 매우 느림 
'''

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from stable_baselines3 import PPO

unity_env = UnityEnvironment(file_name="C:\\Users\\finel\\ml-agents\\rl1.exe")
env = UnityToGymWrapper(unity_env, uint8_visual=True)

obs = env.reset()
print(obs)

model = PPO('MlpPolicy', env, verbose=1) # DQN 모델 불러오기
model.learn(total_timesteps=10000)  

env.close()
