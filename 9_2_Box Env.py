'''
CarRacing으로 Box2D 환경을 살펴봅시다. 
gym 0.21로 진행합니다. 
'''

''' 
환경 불러오고 살펴보기
'''
import gym

env = gym.make('CarRacing-v0')
obs = env.reset()
env.render()
env.close()

'''
환경 살펴보기
'''
import matplotlib.pyplot as plt
plt.imshow(obs) # numpy 배열을 이미지 출력, (W, H, C)

env.action_space  # [방향, 엑셀, 브레이크]
env.action_space.sample()

'''
랜덤 샘플링으로 출력해보기
'''

for i in range(1000):
    obs = env.action_space.sample()  # 랜덤  액션 지정
    env.step(obs)
    env.render()
env.close()
