import gym
import pandas_datareader as pdr
import numpy as np

class stock_env(gym.Env):
    '''
    입력 형태 예시
    stock_code: '005930' 
    start: '2022-01-01'
    end: '2022-10-01'
    '''
    stock_code=None # 변수로 메서드 밖에 선언 시 바로 호출 가능
    start = None
    end = None
    df = None
    cursor = 0
    rewards = 0
    
    # gym.Env에 필요한 변수
    action_space = gym.spaces.Discrete(2)  # 0: 포지션 없음, 1: 매수 포지션
    observation_space = None
    
    def __init__(self, stock_code, start=None, end=None, obs_period = 60):
        super().__init__()
        
        # 데이터 불러오기
        naver = pdr.naver.NaverDailyReader(stock_code, start=start, end=end,  
                                        adjust_price=True)
        df = naver.read()
        df = df.astype('int')  # 정수로 변환
        
        df['date'] = df.index
        df.index = range(len(df))
        
        # 클래스 내에서 활용할 변수 저장
        self.df = df
        self.stock_code = stock_code
        self.start = start
        self.end = end
        self.obs_period = obs_period        
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_period,),
                                                dtype=np.float32)
                                    # 수익률은 -100% ~100% 사이라 가정
        self.buy_price = None                            
        
        
    def plot_price(self):
        '''
        데이터 확인을 위해 추가
        '''
        self.df.Close.plot()
        
    def _get_obs(self): 
        '''
        과거 period 기간을 obs로 넣어 줌, 
        시작점을 1로 만들어서 표준화
        '''
        start_index = self.cursor-self.obs_period
        end_index = self.cursor
        return self.df.Close[start_index:end_index]/self.df.Close[start_index]
    
    def _get_price(self, offset=0): # 현재 가격 가져오기, 매매 저장
    
        return self.df.Close[self.cursor- offset]/self.df.Close[self.cursor-self.obs_period]
        
    def reset(self):
        self.cursor = self.obs_period # 기간 초기화, 어제까지의 데이터로 오늘 액션 결정
        self.rtn = 0  # 수익률 초기화
        self.action_prev = 0  # 바로 전단계 액션을 저장

        self.rewards = 0
        print('reset')
        self.render()
        return self._get_obs()
    
    def step(self, action):
        '''
        action에 따른 행동
        0 --> 0 계속 투자 안함: reward: 0%
        0 --> 1 새로 매수: 매수가 저장
        1 --> 1 계속 보유: 평가 수익률 확인
        1 --> 0 청산: 수익률 누적에 더하기
        '''
        reward = 0
        # print(f'action:{action}')
        if self.action_prev ==0 and action ==0: reward = 0
        if self.action_prev ==0 and action ==1: reward = 0
        if self.action_prev ==1 and action ==1: 
            reward = self._get_price()-self._get_price(1) # 하루동안 수익
            
            # print('포지션 유지')
            # print(self._get_price(), self._get_price(1))
        if self.action_prev ==1 and action ==0:  
            reward = self._get_price()-self._get_price(1)              
            # print('청산')
        self.action_prev = action # stablebaselines에서 dictionary로 들어
        self.rewards += reward
        self.cursor += 1
        
        terminated = False
        if self.cursor >= len(self.df): terminated = True
        elif self.rewards<-1000000: terminated = True  #100만원 잃으면 중단
        
        # print(f'terminage:{terminated}')
        # print(self._get_obs())
        
        return self._get_obs(), reward, terminated, {'rewards':self.rewards}        
    
    def render(self):
        print(f'today: {self.df.index[self.cursor]}, rewards: {self.rewards:.4f}, prev_action:{self.action_prev}')
        
if __name__=='main':
    env = stock_env('005930')
    env.reset()
    env._get_obs()











