import numpy as np

class OUNoise():
    
    """
    Ornstein-Uhlenbeck process for temporally
    correlated noise generation
    
    credit: Adopted from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    
    """
    def __init__(self, action_size, 
                 action_low= -1., action_high = 1.,
                 mu=0.0,theta=0.15, max_sigma = 0.2,
                 min_sigma=0.05, decay_period= 5000):
        
        # Initializing process parameters
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_size
        self.low = action_low
        self.high = action_high
        self.reset()
        
    def reset(self):
        
        """
        Resets the state back to the mean
        
        """
        
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        
        """
        Models the evolution of the state
        
        """
        
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        
        """
        Returns the action added with noise based
        the current time step
        
        """
        
        ou_state = self.evolve_state()
        # Decreasing standard deviation after each time step
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)