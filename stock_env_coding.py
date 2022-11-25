import gym
import pandas_datareader as pdr
import numpy as np

class stock_env(gym.Env):
    '''
    
    '''
    action_space = gym.spaces.Discrete(2)  # 0: 포지션 없음, 1: 매수 포지션
    observation_space = None
    
    def __init__(self, stock_code, start=None, end=None, obs_period = 60):
        super().__init__()  # parent class 초기화
        
        naver = pdr.naver.NaverDailyReader(stock_code, 
                                           start=start, end=end,  
                                           adjust_price=True)
        
        df = naver.read()
        df = df.astype('int')  # 정수로 변환
        
        # 날짜를 컬럼으로 추가, 숫자인덱스로 변경
        df['date'] = df.index  # row index --> 'date' 컬럼에 복사
        df.index = range(len(df))  # 숫자 생성하여 row index 입력
                
        self.df = df
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(obs_period,),
                                                dtype=np.float32)
        self.cursor = obs_period # 기간 초기화, 어제까지의 데이터로 오늘 액션 결정
        self.obs_period = obs_period
        
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
        return self._get_obs()
    
    def step(self, action):
        
        return self._get_obs(), reward, terminated, {'rewards':self.rewards}  

    def render(self):
        pass
    
    
    
        
    
        
env = stock_env('005930')
env.df




    
            
    
        