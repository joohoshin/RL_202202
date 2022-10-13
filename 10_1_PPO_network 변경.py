
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

'''
기본 구조를 확인해봅시다
'''
from torchsummary import summary  
summary(model.policy, (3,96,96)) # pytorch는 채널이 앞쪽


'''  구조를 수정해봅시다 
In short: [<shared layers>, 
           dict(vf=[<non-shared value network layers>], 
                pi=[<non-shared policy network layers>])].
'''
# policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])])
policy_kwargs = {'net_arch':[{'pi':[32, 32], 'vf':[32, 32]}]}
model2 = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_cnn2/", 
             policy_kwargs = policy_kwargs )

from torchsummary import summary  
summary(model2.policy, (3,96,96)) # pytorch는 채널이 앞쪽





