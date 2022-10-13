
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

'''
cnn을 적용해보겠습니다. 
gym 0.21에서 진행합니다. 
'''

'''
stable baselines의 DQN은 box2d에 적용되지 않습니다. 
PPO를 불러와서 학습해봅시다
'''
env = gym.make("CarRacing-v0")  # Box2D로 영상을 보고 학습
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./carracing_ppo_cnn/")
model.learn(total_timesteps=10000, tb_log_name="1st")
model.learn(total_timesteps=10000, tb_log_name="2nd", reset_num_timesteps=False)
model.learn(total_timesteps=30000, tb_log_name="3rd", reset_num_timesteps=False)
model.learn(total_timesteps=20000, tb_log_name="4th", reset_num_timesteps=False)
model.learn(total_timesteps=30000, tb_log_name="5th", reset_num_timesteps=False)
model.learn(total_timesteps=30000, tb_log_name="6th", reset_num_timesteps=False)
model.learn(total_timesteps=30000, tb_log_name="7th", reset_num_timesteps=False)
env.close()

''' 
어느 정도 학습되었는지 살펴봅시다
'''
env = gym.make("CarRacing-v0") 
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs.copy(), deterministic=True )  # obs를 그냥 넣으면 오류 발생, bug 인 듯함
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()

model.save("carracing_ppo_cnn")