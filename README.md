# RLToolKit
Reinforcement Learning Toolkit

A work in progress suite of reusable tools for reinforcement learning

TO DO:
  *Models: Actor/Critic, 
  *Advantage Estimate Functions
  *Utility Functions
   -Q Learning
   -A2C
   -PPO
   -NEC
  *Coach handles multiple environments
  
Example usage:

Construct a coach object with

Coach object handles environment, buffer memory object, and optimzier.

 -Environment name
 -Optimizer type
 -Frameskip: number of frames to skip per step
 -Batch size
 -Utility function
 -Preprocessor function (for pixel images)
 
```python
coach = Coach(env="MsPacman-v4",
              loss_fn=reinforce,
              optim=torch.optim.RMSprop,
              n_agents=n_agents,
              flatten=True,
              frameskip=1,
              batch_size=batch_size,
              preprocess_func=preprocess_frame,
              utility_func=reinforce,
              )
```

Construct an agent object

```python
agent = Softmax_Agent(
                          model=torch.nn.Sequential(),
                          head=torch.nn.Sequential(),
                          batch=batch_size,
                          n_agents=n_agents
                    )
```  
      
Run an episode

```python
rewards, episode_steps = coach.run_episode(
            render=True,
            gamma=discount,
            max_steps=2500,
            explore=True
        )
 ```
 
 Update agent parameters
 ```python
     loss = coach.learn(discount)
```
