# Udacity Deep Reinforcement Learning - Collaboration and Competition Project.

## Learning Algorithm

For this project I have used [MADDPG](https://arxiv.org/abs/1706.02275) algorithm, which is based on the DDPG and is extended for the use of multi-agent learning
by having multiple independent "Actor" networks (one for every Agent) and a single "Critic" network, which has access to the the state of all of the "Actors" during 
the training time. We achieve this by concatenating the state from all the agents during the training and using it as the input for the "critic" model 
which is used to model the "rewards" for the Agent actions. This allows for centralized learning and decentralized execution and results in training multiple 
agents that can act in a globally-coordinated way. Our agents don’t need to access the central critic at test time, they act based on their observations in 
combination with their predictions of other agents behaviors. Since the "Critic" model is trained independently form the actors, we can tailor the 
"Actor" reward functions to exhibit traits of cooperation or competition.

## Model Architecture

The model consists of the two separate neural netwroks for actor and the critic and it follows the architecture from original DDPG paper [Continuous control with deep 
reinforcement learning](https://arxiv.org/abs/1509.02971), with th addition of a single Batch Normalisation layer for both the Actor and the Critic, 
which proved to help with the training speed and stability.

### Actor Model

The actor model is a simple feed forward neural network with 3 fully connected (FC) layers followed by a ReLU activation function. 
Th first layer is also followed by the batch normalisation layer. The final FC layer has an output dimension corresponding to the action size, which is 
transformed by the TanH activation function in order to scale the output from -1 to 1.

The model Layers are as following:
```
input layer:  in 24  out 256
batch norm: 256
ReLU
hidden layer: in 256  out 256
ReLU
output layer: in 256 out 2
TanH
```

### Critic Model

The Critic model is also a feed forward neural network with 3 fully connected (FC) layers. We diverge from the original paper by concatenating the states and actions 
from the actors in the first layer. In Addition the input for the critic has to account for states and actions from multiple agents.

The model Layers are as following:

```
input layer:  in (24 + 2) * 2 out 256
batch norm: 256
ReLU
hidden layer: in 256  out 256
ReLU
output layer: in 256 out 1
```

### Hyper parameters

```
NUM_UPDATES = 1         # number of updates
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
LEARN_EVERY = 1         # learn every # iterations
NOISE_LEVEL = 100000    # number of iterations before removing noise 
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.000    # L2 weight decay
NOISE_LEVEL = 100000    # number of iterations before removing noise 
```

## Results

The agent slowly improves until episode 2000 then experiences a slight dip in performace only to recover around episode 2800 and solving the task by the episode 2881

![performance](results.png)

## Ideas for Future Work

To improve the performance we could try several things:

 - Prioritized experience replay: giving higher sample weights to the experiences that had higher error rates.
 - Try using different architectures for our Actor/Critic models
 - We can switch to a different algorithm like AlfaZero.
 