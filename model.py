import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_layers

class Actor (nn.Module):
    
    """
    Defines the neural network architecture
    that will be used by the actor
    
    """
    
    
    def __init__(self, state_size, action_size, 
                 hidden_neurons, dropout = 0.1):
        
        """
        Arguments: state_size [Int] (State size),
                   action_size [Int] (Action size),
                   hidden_neurons [List] (Neurons in each hidden layer),
                   dropout [Float from 0 to 1] (Dropout Regularization),
                  
        For example,
        
        agent = Actor(state_size = 33,
                     action_size = 4,
                     hidden_neuron = [128, 64],
                     dropout = 0.1)
                   
        This will create a Actor object which will have
        4 layers: the input layer with 33 neurons; next one hidden layer
        with 128 neurons; next another hidden layer with 64 neurons; final
        output layer with 4 neurons. Except the last layer each layer uses 
        relu as activation and dropout regularization. The last layer uses
        tanh activation to bound the outputs between -1 and 1.
        
        """
        
        super().__init__()
    
        
        # Creating sequence of linear layers
        # that will be applied to input
        
        self.layer1 = nn.Linear(state_size, hidden_neurons[0])
        
        self.layer2 = nn.Linear(hidden_neurons[0], hidden_neurons[1])
        
        self.layer3 = nn.Linear(hidden_neurons[1], action_size)
    
        # Dropout value
        self.dropout = dropout
        
        # Initialize the weights
        self.reset_parameters()
        
        
    def reset_parameters(self):
        
        """
        Initializes the weight of the network
        based on the DDPG paper
        https://arxiv.org/abs/1509.02971
        
        """
        
        self.layer1.weight.data.uniform_(*init_layers(self.layer1))
        self.layer2.weight.data.uniform_(*init_layers(self.layer2))
        self.layer3.weight.data.uniform_(-3e-3, 3e-3)

        
    def forward(self, x):
        
        """
        Arguments: x [Torch Tensor] (state)
        
        Returns:  x [Torch Tensor] (action)
        
        """
        
        # Passing the input through the linear 
        # layers with activations and dropout
        # except the last layer 
        
        x = nn.ReLU()(self.layer1(x))
        
        x = nn.Dropout(self.dropout)(x)
        
        x = nn.ReLU()(self.layer2(x))
        
        x = nn.Dropout(self.dropout)(x)
        
        x = torch.tanh(self.layer3(x))
        
        return x
    
    
class Critic (nn.Module):
    
    """
    Defines the neural network architecture
    that will be used by the critic
    
    """
    
    
    def __init__(self, state_size, action_size, 
                 hidden_neurons, dropout = 0.1):
        
        """
        Arguments: state_size [Int] (State size),
                   action_size [Int] (Action size),
                   hidden_neurons [List] (Neurons in each hidden layer),
                   dropout [Float from 0 to 1] (Dropout Regularization),
                   tanh [Bool] (True for actor to clip actions between -1 to 1)
                   
        For example,
        
        agent = Critic(state_size = 33,
                       action_size = 4,
                       hidden_neuron = [128, 64],
                       dropout = 0.1)
                   
        This will create a model object which will have
        4 layers: the input layer with 33+4 = 37 neurons; next one hidden layer
        with 128 neurons; next another hidden layer with 64 neurons; final
        output layer with 1 neuron.The number of hidden layers and neurons
        in each layer is derived from the length of the list and corresponding
        elements respectively. Except the last layer each layer uses relu as 
        activation and dropout regularization. 
        
        """
        
        
        super().__init__()
        
                
        # Creating sequence of linear layers
        # that will be applied to input
        
        self.layer1 = nn.Linear(state_size + action_size, hidden_neurons[0])
        
        self.layer2 = nn.Linear(hidden_neurons[0], hidden_neurons[1])
        
        self.layer3 = nn.Linear(hidden_neurons[1], 1)

        
        # Dropout value
        self.dropout = dropout
        
        # Initializing network weights
        self.reset_parameters()
        
    def reset_parameters(self):
        
        """
        Initializes the weight of the network
        based on the DDPG paper
        https://arxiv.org/abs/1509.02971
        
        """
        
        self.layer1.weight.data.uniform_(*init_layers(self.layer1))
        self.layer2.weight.data.uniform_(*init_layers(self.layer2))
        self.layer3.weight.data.uniform_(-3e-3, 3e-3)

        
    def forward(self, states, actions):
        
        """
        Arguments: states [Torch Tensor],
                   actions [Torch Tensor]
        
        Returns:  x [Torch Tensor] (Q value)
        
        """
        
        # Concatenates states and action
        
        x = torch.cat([states, actions], dim = -1)
        
        # Passing the input through the linear 
        # layers with activations and dropout
        # except the last layer
        
        x = nn.ReLU()(self.layer1(x))
        
        x = nn.Dropout(self.dropout)(x)
        
        x = nn.ReLU()(self.layer2(x))
        
        x = nn.Dropout(self.dropout)(x)
        
        x = self.layer3(x)
                
        return x
    
    
