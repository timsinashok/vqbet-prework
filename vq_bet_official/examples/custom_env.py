import gym
from gym import spaces
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

class RobotEnv(gym.Env):
    """OpenAI Gym wrapper for the bimanual robot environment"""
    
    def __init__(self, task_name='sim_transfer_cube', visual_input=False):
        super().__init__()
        
        # Create the DM Control environment
        if 'sim_transfer_cube' in task_name:
            xml_path = os.path.join(XML_DIR, 'bimanual_viperx_transfer_cube.xml')
            physics = mujoco.Physics.from_xml_path(xml_path)
            task = TransferCubeTask(random=False)
        elif 'sim_insertion' in task_name:
            xml_path = os.path.join(XML_DIR, 'bimanual_viperx_insertion.xml')
            physics = mujoco.Physics.from_xml_path(xml_path)
            task = InsertionTask(random=False)
        else:
            raise NotImplementedError
            
        self.env = control.Environment(
            physics, task, time_limit=20, 
            control_timestep=DT, n_sub_steps=None, 
            flat_observation=False
        )
        
        self.visual_input = visual_input
        
        # Define action space: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0, 
            shape=(14,),
            dtype=np.float32
        )
        
        # Define observation space
        if visual_input:
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, 480, 640),
                    dtype=np.uint8
                ),
                'desired_goal': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),  # Adjust based on your goal space
                    dtype=np.float32
                ),
            })
        else:
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(28,),  # qpos(14) + qvel(14)
                    dtype=np.float32
                ),
                'desired_goal': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),  # Adjust based on your goal space
                    dtype=np.float32
                ),
            })

        self.current_goal = None

    def reset(self):
        """Reset the environment and return initial observation."""
        time_step = self.env.reset()
        
        if self.visual_input:
            obs = {
                'image': time_step.observation['images']['vis'],
                'desired_goal': self.current_goal if self.current_goal is not None 
                               else np.zeros(7, dtype=np.float32)
            }
        else:
            obs = {
                'observation': np.concatenate([
                    time_step.observation['qpos'],
                    time_step.observation['qvel']
                ]),
                'desired_goal': self.current_goal if self.current_goal is not None 
                               else np.zeros(7, dtype=np.float32)
            }
        return obs

    def step(self, action):
        """Execute action and return new observation, reward, done, info."""
        time_step = self.env.step(action)
        
        if self.visual_input:
            obs = {
                'image': time_step.observation['images']['vis'],
                'desired_goal': self.current_goal if self.current_goal is not None 
                               else np.zeros(7, dtype=np.float32)
            }
        else:
            obs = {
                'observation': np.concatenate([
                    time_step.observation['qpos'],
                    time_step.observation['qvel']
                ]),
                'desired_goal': self.current_goal if self.current_goal is not None 
                               else np.zeros(7, dtype=np.float32)
            }
        
        reward = time_step.reward
        done = time_step.last()
        info = {
            'env_state': time_step.observation['env_state'],
            'success': reward == self.env.task.max_reward
        }
        
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        """Render the environment."""
        if mode == 'rgb_array':
            return self.env.physics.render(height=480, width=640, camera_id='front_close')
        return None

    @staticmethod
    def get_goal_fn(obs_dict):
        """Required function for goal-conditioned tasks."""
        return obs_dict['desired_goal']

    def set_goal(self, goal):
        """Set the current goal for the environment."""
        self.current_goal = goal

# Register the environment
from gym.envs.registration import register

register(
    id='BimanualRobot-v0',
    entry_point='robot_env:RobotEnv',
    max_episode_steps=280,
)