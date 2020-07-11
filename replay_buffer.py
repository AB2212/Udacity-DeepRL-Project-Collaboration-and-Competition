import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from collections import deque, namedtuple

#Device type (use cuda if gpu is available else cpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    
    """
    Replay Buffer class
    
    """ 
    
    def __init__(self, buffer_size, batch_size, seed=0):
        
        """
        Arguments: buffer_size [Int] (max experience that will be stored,
                   past experiences will be deleted to store new ones),
                   batch_size [Int] (Batch Size for training)
        """
        
      
        # Memory for storing experiences
        self.memory = deque(maxlen = buffer_size)
        
        self.batch_size = batch_size
        
        self.seed = random.seed(seed)
        
        # Storing Experiences as named tuple
        self.experiences =  namedtuple('Experience', 
                                       ['state', 'action', 'reward', 'next_state', 'done'])
    
    
    def add(self, state, action, reward, next_state, done):
        
        """
        Arguments: state [Numpy Array] (current state),
                   action [Numpy Array] (action taken at current state)
                   reward  (float) (reward received for current action)
                   next_state [Numpy Array] (next state as a result of current action)
                   done (Bool) (Whether episode has end or not)
                   
        Description: Adds experience to memory
        
        """
        
        self.memory.append(self.experiences(state, action, 
                                           reward, next_state, done))

        
    def sample(self):
        
        """
        Returns: Tuple of Torch Tensors containing
                 States, Actions, Rewards, Next States, Dones
                 of specified batch size
        
        """
        
        # Randomly sampling number of experiences equal to batch size
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Stacking the values and converting to tensors
        # to feed in batch
        
        states = (torch.from_numpy(np.vstack([e.state 
                                             for e in experiences if e is not None ]))
                  .float().to(device))
        
        actions = (torch.from_numpy(np.vstack([e.action
                                              for e in experiences if e is not None]))
                   .float().to(device))
        
        rewards = (torch.from_numpy(np.vstack([e.reward 
                                              for e in experiences if e is not None]))
                   .float().to(device))
        
        next_states = (torch.from_numpy(np.vstack([e.next_state 
                                                  for e in experiences if e is not None]))
                       .float().to(device))
                
        dones = (torch.from_numpy(np.vstack([e.done 
                                            for e in experiences if e is not None])
                                 .astype(np.uint8))
                    .float().to(device))
        

  
        return (states, actions, rewards, next_states, dones)
        
    
    def __len__(self):
        
        """
        Dunder Function to return
        length of memory
        
        """
        
        return len(self.memory)
        
