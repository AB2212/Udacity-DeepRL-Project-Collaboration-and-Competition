class Config():

    BUFFER_SIZE = int(1e6)       # replay buffer size
    BATCH_SIZE = 512             # minibatch size 
    UPDATE_EVERY = 1             # how often to update the network
    NUM_UPDATES = 2              # Number of passes 
    NUM_NEURONS_LAYER1 = 512     # Number of neurons in layer 1 
    NUM_NEURONS_LAYER2 = 256     # Number of neurons in layer 2
    learning_rate_actor = 1e-4   # Learning rate for actor
    learning_rate_critic = 1e-4  # Learning rate for critic
    tau = 0.2                    # tau fo soft update
    gamma = 0.99                 # discounting factor



    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")