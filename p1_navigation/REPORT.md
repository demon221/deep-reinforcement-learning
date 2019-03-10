# Project1: Navigation

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: ./dqn-algorithm.png "DQN Alogorithm"
[image3]: ./dqn-training-episodes.png "DQN Training Episode"
[image4]: ./dqn-scores.png "DQN Scores"

### Report

This report details the methods and algorithms I used to solve the Navigation project as part of the Deep Reinforcement Learning Nanodegree from Udacity.

The goal of this project is to train an agent with deep reinforcement learning algorithms so that it can collect as many yellow bananas and avoid as many blue bananas as possible in a large, square world.

![image1]

### Project Structure
This folder contains:
 - `Navigation.ipynb` Jupyter Notebook that has the solution. 
 - `dqn_network.py` consists of Neural Network definition.
 - `dqn_agent.py` contains the DQN agent. 
 - `dqn_checkpoint.pth` is the weights of the neural network of the trained DQN.

### DQN algorithm
We trained an agent using a simplified version of deep Q-network (DQN) algorithm. This algorithm uses a replay memory that has a finite capacity and works in FIFO way. DQN algorithm also uses local and target action-value functions q and q'. These functions are implemented as feedforward neural networks and have identical architecture. The networks take as input a state vector and output action value of every possible action. We denote by q(S,w,A) the value of action A at state S, computed by local neural network with weights w. Similarly, q'(S,w',A) is the value of action A at state S, computed by target neural network with weights w'. An ε-greedy policy based on q(S,w,·) chooses a random action with probability ε and an action that maximizes q(S,w,·) with probability 1-ε. 

![image2]

#### Neural Network Architecure
- Input Layer: 37
- Hidden Layer 1: 64
- Hidden Layer 2: 64
- Output Layer: 4

#### Hyperparameters
- BUFFER_SIZE = int(1e5)  
- BATCH_SIZE = 64         
- GAMMA = 0.99           
- TAU = 1e-3             
- LR = 5e-4              
- UPDATE_EVERY = 4     

#### Results   
Number of episodes needed to solve the environment : `862`

![image3]

Scores for number of episodes

![image4]

Below is the testing result for 10 episodes with trained model weights:

- Episode 1: 11.0
- Episode 2: 22.0
- Episode 3: 11.0
- Episode 4: 18.0
- Episode 5: 16.0
- Episode 6: 14.0
- Episode 7: 17.0
- Episode 8: 16.0
- Episode 9: 19.0
- Episode 10: 14.0

All the 10 episodes scores are: 
`[11.0, 22.0, 11.0, 18.0, 16.0, 14.0, 17.0, 16.0, 19.0, 14.0]`

The Mean Score is: `15.8`

#### Future Work
The agent can be further improved by the following:

* Implementing prioritized experience replay so that the agent can focus more on experience which has larger error.

* Implementing Expected SARSA. For

```
a(t+1)=ArgMax(Q_target(s(t+1),a))
y= r(t)+Gamma*Q_target(s(t+1),a(t+1))
```

instead of only taking account of the maximum Q value in the next state.
```
a1(t+1)=ArgMax(Q_target(s(t+1),a))
a2(t+1)=random(4)
y=r(t)+Gamma*((1-epsilon)*Q_target(s(t+1),a1(t+1)) + epsilon*Q_target(s(t+1),a2(t+1)))
```
we can take epsilon into account. So this will change the algorithm from off-policy to on-policy which can help to stablize the results.

* Implementing Rainbow Algorithm [https://arxiv.org/pdf/1710.02298.pdf] which combines good features from different algorithms to form an itegrated agent.