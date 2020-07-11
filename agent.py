import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Actor, Critic
from utils import init_layers
from config_file import Config


BUFFER_SIZE = Config.BUFFER_SIZE                # replay buffer size
BATCH_SIZE = Config.BATCH_SIZE                  # minibatch size 
UPDATE_EVERY = Config.UPDATE_EVERY              # how often to update the network
NUM_UPDATES = Config.NUM_UPDATES                # Number of passes

#Device type (use cuda if gpu is available else cpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    
    def __init__(self, state_size, action_size, 
                 num_agents,
                 hidden_neurons, dropout = 0.1,
                 gamma = 0.99, 
                 learning_rate_actor = 1e-4, 
                 learning_rate_critic = 1e-4,
                 seed = 0, tau = 1e-3):
        
        """
        Arguments: state_size [Int] (State size),
                   action_size [Int] (Number of Actions),
                   hidden_neurons [List] (Neurons in each hidden layer),
                   dropout [Float from 0 to 1] (Dropout Regularization),
                   gamma [Float 0 to 1] (Discounting factor),
                   learning_rate_actor [Float 0 to 1] (Learning rate for weight 
                                                       update for actor),
                   learning_rate_critic [Float 0 to 1] (Learning rate for weight
                                                        update for critic),
                   seed [Int] (random seed),
                   tau [Float from 0 to 1] (Soft update rate for target DQN)
                    
        """
    
        
        super().__init__()
        
        
        # Initializing main actor 
        self.actor_main = Actor(state_size, action_size, 
                              hidden_neurons, dropout).to(device)
        
        # Initializing target actor 
        self.actor_target = Actor(state_size, action_size, 
                              hidden_neurons, dropout).to(device)
        
        # Initializing main critic
        self.critic_main = Critic(state_size*num_agents, action_size*num_agents, 
                              hidden_neurons, dropout).to(device)
        
        # Initializing target critic
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, 
                              hidden_neurons, dropout).to(device)
        
        # Optimizer for actor
        self.optimizer_actor = optim.Adam(self.actor_main.parameters(),
                                          lr = learning_rate_actor)
        
        # Optimizer for critic
        self.optimizer_critic = optim.Adam(self.critic_main.parameters(),
                                          lr = learning_rate_critic)
        
        # Weight for soft update
        self.tau = tau

        self.hard_update(self.actor_main, self.actor_target)
        
        self.hard_update(self.critic_main, self.critic_target)
        
  
    def act(self, state):
        
        """
        Arguments: state [Torch Tensor] (environment state),
                   
                   
        Returns: Actions [Numpy Array]
        
        """
        
        # Setting the network to evaluation mode
        self.actor_main.eval()
        
        # Using torch no grad as action selection
        with torch.no_grad():
        
            state = torch.from_numpy(state).float().to(device)
            
            action =  self.actor_main(state)
        
            action.clamp_(-1., 1.)
        
        # Setting model for training
        self.actor_main.train()
        
        return action.squeeze(0).cpu().numpy()
    
    
        
    def soft_update(self, main_model, target_model):
        
        """
        Arguments: main_model [MLP object]
                   target_model [MLP object]
                   
        Description: updates the weight of the target model 
                     network using weighted sum of previous target model  
                     parameters and current main model parameters 
        """
        
        for target_param, main_param in zip(target_model.parameters(),
                                             main_model.parameters()):
            
            target_param.data.copy_(self.tau * main_param.data + (1.-self.tau) * target_param.data)
            
    
    def hard_update(self, main_model, target_model):
        
        """
        Arguments: main_model [MLP object]
                   target_model [MLP object]
                   
        Description: updates the weight of the target model 
                     network using main model parameters 
        """
        
        for target_param, main_param in zip(target_model.parameters(),
                                             main_model.parameters()):
            
            target_param.data.copy_(main_param.data)
    