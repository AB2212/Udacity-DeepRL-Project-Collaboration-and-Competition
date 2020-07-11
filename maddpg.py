import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from agent import Agent
from noise_generator import OUNoise
from replay_buffer import ReplayBuffer

class MADDPG(nn.Module):
    
    def __init__(self, state_size, action_size,
                 num_agents, config):
        """
        Arguments: state_size [Int] (State size),
                   action_size [Int] (Number of Actions),
                   num_agents [Int] (Number of Agents),
                   config [Config] (config class containing all
                                    the hyperparameters)
                    
        """
        
        self.state_size = state_size
        
        self.action_size = action_size
        
        self.config = config
        
        self.num_agents = num_agents
        
        # Initializing the agents
        self.agents = [Agent(state_size, action_size,
                             num_agents,
                             [config.NUM_NEURONS_LAYER1,
                              config.NUM_NEURONS_LAYER2],
                             learning_rate_actor= config.learning_rate_actor, 
                             learning_rate_critic= config.learning_rate_critic,
                             tau = config.tau)
                       
                       for _ in range(num_agents)]
        

        self.memory = ReplayBuffer(config.BUFFER_SIZE,
                                   config.BATCH_SIZE,
                                   seed = 0)
        
        self.t_step = 0
        
        self.gamma = config.gamma
        
            
        self.noises = [OUNoise(action_size) for _ in range(self.num_agents)]
        
        
        
        
    def act(self, state, step = 0):
        
        """
        Arguments: state [Torch Tensor] (environment state),
                   
                   
        Returns: Actions [Numpy Array]
        
        """
        if step==0:
            
            for noise in self.noises:
                noise.reset()
    
        # Using torch no grad as action selection
        actions = [agent.act(state) 
                   for agent,state in zip(self.agents, state)]
        
        actions = [noise.get_action(action, step) 
                      for action, noise in zip(actions, self.noises)]
        
        
        return np.array(actions)
    
    
    def step(self,states, actions, rewards, next_states, dones):
        
        """
        Arguments: states [Numpy Array] (current state),
                   actions [Numpy Array] (action taken at current state)
                   rewards  (float) (reward received for current action)
                   next_states [Numpy Array] (next state as a result of current action)
                   dones (Bool) (Whether episode has end or not)
        
        """
        
        # Adding experience to replay buffer
        
 
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # Keeping track of time step
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        
        # Updating model weights after 
        # regular intervals
        if self.t_step == 0:
            
            # Checking if enough samples are present
            if len(self.memory) > self.config.BATCH_SIZE:
                
                for _ in range(self.config.NUM_UPDATES):
                
                    self.learn()
                    
    def learn(self):
        
        """
        Updates the parameters of the agent
        
        """
        
    
        for i,agent in enumerate(self.agents):
            
            
            # Sampling from replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample()
        

            # Setting target network to evaluaion mode
            agent.critic_target.eval()
            agent.actor_target.eval()

            
            # Calculating actions for other agents for
            # current and next states, this is needed
            # by the critic
            
            next_action_stack = []
            
            action_stack = []

            for j,agent_j in enumerate(self.agents):

                # Setting network to evaluation mode
                agent_j.actor_target.eval()

                next_action_stack.append(agent_j.actor_target(next_states[:,j,:])
                                         .unsqueeze(1))
                

                agent_j.actor_main.eval()
                
                action = agent_j\
                         .actor_main(states[:,j,:])\
                         .unsqueeze(1)
                
                # Detaching all actions except for
                # current agent
                if i != j:
                    
                    action.detach()
                    
                action_stack.append(action)
                
                
            action_stack = (torch.cat(action_stack, dim = 1)
                            .view(-1, self.action_size*self.num_agents))


            next_action_stack = (torch.cat(next_action_stack, dim = 1) 
                                 .view(-1,self.action_size*self.num_agents))

               
            # Creating target value for critic
            with torch.no_grad():
                        
                Q_next_state = agent.critic_target(next_states
                                                   .view(-1,
                                                    self.state_size*self.num_agents),
                                                   
                                                    next_action_stack)
                

                Q_target = rewards[:,i,:] + self.gamma*Q_next_state*(1.- dones[:,i,:])
                

            # Training Critic
            agent.critic_main.train()

            # Calculating critic Q value for state and action
            Q_critic = agent.critic_main(states.view(-1,self.state_size*self.num_agents),
                                         actions.view(-1, self.action_size*self.num_agents))


            # Using MSE loss for critic
            critic_loss = F.mse_loss(Q_target, Q_critic).mean()

            # Zero grad removes any accumulated gradient
            agent.optimizer_critic.zero_grad()

            # Calculating gradients using backpropagation
            critic_loss.backward()

            # Clipping high gradients
            torch.nn.utils.clip_grad_norm_(agent.critic_main.parameters(), 1)

            # Updating Weights
            agent.optimizer_critic.step()

        

            # Calculating critic Q value for state and 
            # actor's action
                
            agent.actor_main.train()
                
            actor_loss = -agent.critic_main(states.view(-1, 
                                                        self.state_size*self.num_agents),
                                              action_stack).mean()

            # Zero grad removes any accumulated gradient
            agent.optimizer_actor.zero_grad()

            # Calculating gradients using backpropagation
            actor_loss.backward()

            # Clipping high gradients
            torch.nn.utils.clip_grad_norm_(agent.actor_main.parameters(), 1)

            # Updating Weights
            agent.optimizer_actor.step()



            # Updating the target network using soft update through
            # weighted sum of previous parameters and current parameters
            agent.soft_update(agent.actor_main, agent.actor_target)

            agent.soft_update(agent.critic_main, agent.critic_target)



