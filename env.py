import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoviEnv:

    def __init__(self, device="cpu", keys = ("poses", "trans", "betas")):
        # NOTE states and actions as dicts?
        # NOTE how to handle reward, add env-native reward here in the class and if adding GAN-loss this needs to 
        # be added in the RL training loop as reward weights change through training.
        # NOTE: how to handle done logic? 
        self.device = torch.device(device)
        self.keys = keys
        self.done = False

    def reset(self, init_state):

        self.state = {
            "lifted_state" : {
                key: torch.from_numpy(init_state[key]).to(self.device) for key in self.keys
            },
            # NOTE: zeros for corrected state as initial.
            "corrected_state" : {
                key: torch.zeros_like(torch.from_numpy(init_state[key])).to(self.device) for key in self.keys
            }
        }
        
        return self.state
    
    def step(self, action, next_lifted_state, current_gt_state):
        # NOTE: action is dict of same keys as state, with values being the corrections to apply to the lifted state
        # NOTE: how to handle actions for betas? should probably not happen at every time step, once per clip..., 
        # handle separately completely? 
        for key in self.keys:
            self.state["corrected_state"][key] = self.state["lifted_state"][key] + action[key]
            self.state["lifted_state"][key] = torch.from_numpy(next_lifted_state[key]).to(self.device)
        
        self.done = False # NOTE: placeholder, add done logic for early stopping e.g. pelvis height....
        
        reward = self.env_reward(self.state["corrected_state"], current_gt_state)
        return self.state, reward, self.done

    def env_reward(self, corrected_state, gt_state):
        # NOTE: placeholder simple mse loss.
        reward = 0
        for key in self.keys:
            reward -= F.mse_loss(corrected_state[key], torch.from_numpy(gt_state[key]).to(self.device))
        return reward
