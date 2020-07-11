import numpy as np
import matplotlib.pyplot as plt

def init_layers(layer):
    
    """
    Function to calculate the initialization
    values for the weights of the network,
    based on the DDPG paper
    https://arxiv.org/abs/1509.02971
    
    """
    
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    
    return (-lim, lim)

def calc_moving_average(scores, length = 100):
    
    """
    Calculates moving average of scores
    
    """
    
    moving_average = []
    
    for i in range(len(scores)):

        if i<length:
            mean = np.mean(scores[:i+1])

        else: 
            mean = np.mean(scores[i-length+1 : i+1])

        moving_average.append(mean)
        
    return np.array(moving_average)
    

def plot_reward(scores,
                filename = 'reward_plot.png'):
    """
    Plots scores and moving average of it
    
    """

    moving_average = calc_moving_average(scores)
    plt.plot( scores, label = 'score' )
    plt.plot( moving_average, label = 'moving average')
    plt.title("Reward at each episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.savefig(filename)
    plt.show()
    
