
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

'''
PPO의 cnnpolicy를 변경해봅시다. 
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
'''
env = gym.make("CarRacing-v0")  # Box2D로 영상을 보고 학습
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_cnn/")
obs = env.reset()
env.close()

''' CNN을 살펴보자
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
'''

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import is_image_space

class CustomCNN(NatureCNN):  

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),   # 한 층을 추가해보았음
            nn.ReLU(),
            nn.Flatten(),
        )
 
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
policy_kwargs = {'net_arch':[{'pi':[32, 32], 'vf':[32, 32]}],
                 'features_extractor_class':CustomCNN}
model3 = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_cnn2/", 
             policy_kwargs = policy_kwargs )
print(model3.policy)

from torchsummary import summary  
summary(model3.policy, (3,96,96)) 

model3.learn(total_timesteps=10000, tb_log_name="1st")  
model3.learn(total_timesteps=100000, tb_log_name="2nd", reset_num_timesteps=False)       

''' 학습된 내용을 살펴봅시다
'''

env = gym.make("CarRacing-v0") 
obs = env.reset()
for i in range(1000):
    action = model.predict(obs.copy(),deterministic=True )  # obs를 그냥 넣으면 오류 발생, bug 인 듯함
    obs, rewards, dones, info = env.step(action[0])
    env.render()
env.close()

model.save("carracing_ppo_cnn")