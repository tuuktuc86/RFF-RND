import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from .base import BaseAlgorithm

class SAC(BaseAlgorithm):
    def __init__(self, 
                 actor, 
                 critic, 
                 uncertainty_module, 
                 cfg, 
                 device):
        super().__init__(cfg, device)
        
        self.actor = actor
        self.critic = critic
        self.critic_target = copy_model(critic)
      
        self.uncertainty_module = uncertainty_module
        self.device = device
        
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        
        self.lambda_actor = cfg.lambda_actor
        self.lambda_critic = cfg.lambda_critic

        self.automatic_entropy_tuning = cfg.get('automatic_entropy_tuning', True)
        
        if self.automatic_entropy_tuning:
            if cfg.target_entropy == 'auto':
                self.target_entropy = -float(actor.mu_head.out_features)
            else:
                self.target_entropy = cfg.target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(cfg.alpha).to(device)

        self.actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.lr_actor, fused=True)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.lr_critic, fused=True)

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + local_param.data * self.tau
            )

    def train_rnd_only(self, batch):
       
        logs = self.uncertainty_module.update(batch)
        

        if 'loss' in logs:
            return logs['loss']
        elif 'rnd_loss' in logs:
            return logs['rnd_loss']
        elif 'uncertainty_loss' in logs:
            return logs['uncertainty_loss']
        return -1.0

    def update(self, batch, log_this_step=False):
        logs = {}
        

        if log_this_step:
            with torch.no_grad():
                obs = batch['obs']
                action = batch['action']
                unc_val = self.uncertainty_module.compute_uncertainty(obs, action)
                logs['uncertainty_mean'] = unc_val.mean().item()

        obs = batch['obs']
        action = batch['action']
        reward = batch['reward']
        next_obs = batch['next_obs']
        done = batch['done']


        new_action, log_prob, _ = self.actor.sample(obs)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
            if log_this_step:
                logs['alpha_loss'] = alpha_loss.item()
                logs['alpha'] = self.alpha.item()
        else:
            if log_this_step:
                logs['alpha'] = self.alpha.item()

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_action)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            
            if self.lambda_critic > 0:
                unc_score = self.uncertainty_module.compute_uncertainty(next_obs, next_action)
                min_q_target -= self.lambda_critic * unc_score

            target_q = reward + (1 - done) * self.gamma * min_q_target

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if log_this_step:
            logs['critic_loss'] = critic_loss.item()


        new_action, log_prob, _ = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - min_q_new).mean()
        
        if self.lambda_actor > 0:
            unc_score_actor = self.uncertainty_module.compute_uncertainty(obs, new_action)
            actor_loss += self.lambda_actor * unc_score_actor.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if log_this_step:
            with torch.no_grad():
                avg_reward = reward.mean().item()
                logs['diagnostics/avg_reward'] = avg_reward


                unc_val = self.uncertainty_module.compute_uncertainty(obs, action)
                avg_unc = unc_val.mean().item()
                avg_unc = (avg_unc + 1e-8) ** 0.5
                logs['diagnostics/avg_uncertainty'] = avg_unc

                actual_penalty = self.lambda_critic * avg_unc
                logs['diagnostics/penalty_magnitude'] = actual_penalty

                if abs(avg_reward) > 1e-6:
                    logs['diagnostics/penalty_ratio'] = abs(actual_penalty / avg_reward)
                else:
                    logs['diagnostics/penalty_ratio'] = 0.0

        self._soft_update(self.critic, self.critic_target)
        
        return logs

def copy_model(model):
    return copy.deepcopy(model)