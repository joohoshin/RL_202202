'''
https://wegonnamakeit.tistory.com/59 
'''

import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32
target_update_interval = 10
n_step = 100000
exploration_fraction = 0.05
exploration_initial_eps = 0.08
exploration_final_eps = 0.01
min_buffer_size = 2000


# ReplayBuffer 만들기
class ReplayBuffer():
    
    # 버퍼 사이즈 만큼 메모리 만들기
    def __init__(self):
        # deque는 FIFO(First In First Out) 저장, 리스트와 유사하나 읽은 건 사라짐
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    # 버퍼에 저장
    def put(self, transition):
        self.buffer.append(transition)
    
    # 버퍼에서 샘플 데이터 추출
    def sample(self, n):
        
        # 버퍼에서 샘플 추출
        mini_batch = random.sample(self.buffer, n)
        states, actions, rewards, state_t1s, dones = [], [], [], [], []
        
        # 리스트에 배치에 대한 output 저장
        for transition in mini_batch:
            state, action, reward, state_t1, done = transition
            states.append(state)
            actions.append([action])
            rewards.append([reward])
            state_t1s.append(state_t1)
            dones.append([done])

        # 저장된 값을 출력
        return torch.tensor(states, dtype=torch.float), torch.tensor(actions), \
               torch.tensor(rewards), torch.tensor(state_t1s), \
               torch.tensor(dones)
    
    # 길이 출력 함수
    def __len__(self):
        return len(self.buffer)

# Q network
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    # Action을 결정    
    def sample_action(self, obs, epsilon):
        
        out = self.forward(obs) # 모델에서 액션 선택
        coin = random.random() # 랜덤 선택
        
        # epsilon 보다 작은 경우에는 랜덤으로 선택, 큰 경우에는 모델의 아웃풋을 선택
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()  # 가장 높은 확률 값으로 액션 선택

# Training            
def train(q, q_target, memory, optimizer):
    
    # update_interval 기간 동안 훈련
    for i in range(target_update_interval):
        state, action, reward, state_t1, done = memory.sample(batch_size)
        
        # q_network에서 out 출력
        q_out = q(state)
        q_a = q_out.gather(1,action)
        max_q_prime = q_target(state_t1).max(1)[0].unsqueeze(1)
        
        # loss 계산
        target = reward + gamma * max_q_prime * done
        loss = F.smooth_l1_loss(q_a, target)
        
        # target network 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 스크립트 시작

# 환경 불러오기
env = gym.make('CartPole-v1')

# 2개의 q-network 만들기
q = Qnet()
q_target = Qnet()

# q_target network를 q-network와 동일하게 셋팅
q_target.load_state_dict(q.state_dict())

# 초기화
memory = ReplayBuffer()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
score = 0.0  

# 학습 시작하기
for n_epi in range(n_step):
    
    # randon 비중 선형 감소
    epsilon = max(exploration_final_eps , 
                  exploration_initial_eps + (n_epi/n_step) * \
                      (exploration_final_eps - exploration_initial_eps) / exploration_fraction) 
    done = False
    s = env.reset()    
    
    # 에피소드별 반복
    while not done:  
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        
        # 진행상황 버퍼 저장
        memory.put((s,a,r/100.0,s_prime, done_mask))
        s = s_prime
        
        # 스코어 계산
        score += r
        if done:
            break
        
    # 버퍼 사이즈가 일정 수준 이상일 때만 훈련    
    if len(memory)>min_buffer_size:
        train(q, q_target, memory, optimizer)

    # target_update를 진행 interval 기간마다
    if n_epi % target_update_interval==0 and n_epi!=0:
        q_target.load_state_dict(q.state_dict())
        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            n_epi, score/target_update_interval, len(memory), epsilon*100))
        
        # 스코어 초기화
        score = 0.0
env.close()
