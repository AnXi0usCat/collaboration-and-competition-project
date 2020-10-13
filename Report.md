# Udacity Deep Reinforcement Learning - Collaboration and Competition Project.

## Learning Algorithm

For this project I have used [MADDPG](https://arxiv.org/abs/1706.02275) algorithm, which is based on the DDPG and is extended for the use of multi-agent learning
by having multiple independent "Actor" networks (one for every Agent) and a single "Critic" network, which has access to the the state of all of the "Actors" during 
the training time. We achieve this by concatenating the state from all the agents during the training and using it as the input for the "critic" model 
which is used to model the "rewards" for the Agent actions. This allows for centralized learning and decentralized execution and results in training multiple 
agents that can act in a globally-coordinated way. Our agents don’t need to access the central critic at test time, they act based on their observations in 
combination with their predictions of other agents behaviors and since the "Critic" model is trained independently form the actors, we can tailor the 
"Actor" reward functions to exhibit traits of cooperation or competition.

## Model Architecture

## Results

The agent slowly improves until episode 2000 then experiences a slight dip in performace only to recover around episode 2800 and solving the task by the episode 2881

![performance](results.png)

## Ideas for Future Work

To improve the performance we could try several things:

 - Prioritized experience replay: giving higher sample weights to the experiences that had higher error rates.
 - Try using different architectures for our Actor/Critic models
 - We can switch to a different algorithm like AlfaZero.
 