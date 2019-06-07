import numpy as np
class Evaluate:
    def __init__(self, num_trajectories, approximatePolicy, sampleTrajectory, rewardFunction):
        self.num_trajectories = num_trajectories
        self.approximatePolicy = approximatePolicy
        self.sampleTrajectory = sampleTrajectory
        self.rewardFunction = rewardFunction

    def __call__(self, actor_model):
        # actor_model, critic_model = model

        policy = lambda state: self.approximatePolicy(state, actor_model)
        trajectories = [self.sampleTrajectory(policy) for _ in range(self.num_trajectories)]
        episode_rewards = [np.sum([self.rewardFunction(state, action) for state, action in trajectory]) for trajectory in trajectories]
        benchmark = [np.mean(episode_rewards)]

        return benchmark


    

