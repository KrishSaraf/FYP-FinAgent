"""
FinRL-inspired Deep Reinforcement Learning Agents
Adapted for Indian Stock Market Trading
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gymnasium as gym
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for continuous action spaces
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state)
        return action, value


class PPOAgent:
    """
    Proximal Policy Optimization Agent for stock trading
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        # Initialize networks
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory for experience replay
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, value = self.policy(state)
            action = action.cpu().numpy()
            value = value.cpu().numpy()
        
        return action, value
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def update(self):
        """Update policy using PPO"""
        if len(self.memory['states']) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(self.memory['states']).to(self.device)
        actions = torch.FloatTensor(self.memory['actions']).to(self.device)
        rewards = torch.FloatTensor(self.memory['rewards']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        dones = torch.BoolTensor(self.memory['dones']).to(self.device)
        
        # Calculate advantages
        advantages = self._calculate_advantages(rewards, old_values, dones)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy outputs
            new_actions, new_values = self.policy(states)
            
            # Calculate new log probabilities
            new_log_probs = -torch.sum((actions - new_actions) ** 2, dim=1) * 0.5
            
            # Calculate ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.memory = {key: [] for key in self.memory.keys()}
    
    def _calculate_advantages(self, rewards, values, dones):
        """Calculate advantages using GAE"""
        advantages = []
        advantage = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                advantage = 0
            advantage = rewards[i] + self.gamma * advantage - values[i]
            advantages.insert(0, advantage)
        
        return torch.FloatTensor(advantages).to(self.device)
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save(self.policy.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model from file"""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent for continuous control
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Actor networks
        self.actor = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize target networks
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        
        # Replay buffer
        self.memory = []
        self.memory_size = 10000
    
    def select_action(self, state, add_noise=True):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor(state)
            if add_noise:
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1, 1)
        
        return action.cpu().numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def update(self, batch_size=64):
        """Update networks using DDPG"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.FloatTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, _ = self.actor_target(next_states)
            target_q = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (~dones).unsqueeze(1)
        
        current_q = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, _ = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, new_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
    
    def _hard_update(self, target, source):
        """Hard update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _soft_update(self, target, source):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


class A2CAgent:
    """
    Advantage Actor-Critic Agent
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.device = device
        
        # Initialize network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, value = self.policy(state)
            action = action.cpu().numpy()
            value = value.cpu().numpy()
        
        return action, value
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def update(self):
        """Update policy using A2C"""
        if len(self.memory['states']) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(self.memory['states']).to(self.device)
        actions = torch.FloatTensor(self.memory['actions']).to(self.device)
        rewards = torch.FloatTensor(self.memory['rewards']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        dones = torch.BoolTensor(self.memory['dones']).to(self.device)
        
        # Calculate advantages
        advantages = self._calculate_advantages(rewards, old_values, dones)
        returns = advantages + old_values
        
        # Get current policy outputs
        new_actions, new_values = self.policy(states)
        new_log_probs = -torch.sum((actions - new_actions) ** 2, dim=1) * 0.5
        
        # Actor loss
        actor_loss = -(new_log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(new_values.squeeze(), returns.detach())
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear memory
        self.memory = {key: [] for key in self.memory.keys()}
    
    def _calculate_advantages(self, rewards, values, dones):
        """Calculate advantages"""
        advantages = []
        advantage = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                advantage = 0
            advantage = rewards[i] + self.gamma * advantage - values[i]
            advantages.insert(0, advantage)
        
        return torch.FloatTensor(advantages).to(self.device)
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save(self.policy.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model from file"""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


def create_agent(
    agent_type: str,
    state_dim: int,
    action_dim: int,
    **kwargs
):
    """
    Factory function to create different types of agents with dynamic network sizing
    
    Args:
        agent_type: Type of agent ("PPO", "DDPG", "A2C")
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        **kwargs: Additional arguments for agent initialization
    
    Returns:
        Configured agent
    """
    # Dynamic network sizing based on state/action dimensions
    if 'hidden_dim' not in kwargs:
        # Scale hidden dimension based on state and action dimensions
        base_hidden = 128
        scale_factor = max(1, (state_dim + action_dim) // 50)
        kwargs['hidden_dim'] = min(base_hidden * scale_factor, 1024)  # Cap at 1024
    
    if agent_type.upper() == "PPO":
        return PPOAgent(state_dim, action_dim, **kwargs)
    elif agent_type.upper() == "DDPG":
        return DDPGAgent(state_dim, action_dim, **kwargs)
    elif agent_type.upper() == "A2C":
        return A2CAgent(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
