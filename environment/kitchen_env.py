import _init_paths
import numpy as np
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from environment.base_env import base_env

class TaskParameters:
    def __init__(self):
        self.ALL_TASK = ['bottom_burner', 'top_burner', 'light_switch', 'slide_cabinet', 'hinge_cabinet', 'microwave', 'kettle']

        self.OBS_ELEMENT_INDICES = {
            'bottom_burner': np.array([11, 12]),
            'top_burner': np.array([15, 16]),
            'light_switch': np.array([17, 18]),
            'slide_cabinet': np.array([19]),
            'hinge_cabinet': np.array([20, 21]),
            'microwave': np.array([22]),
            'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
            }

        self.OBS_ELEMENT_GOALS = {
            'bottom_burner': np.array([-0.88, -0.01]),
            'top_burner': np.array([-0.92, -0.01]),
            'light_switch': np.array([-0.69, -0.05]),
            'slide_cabinet': np.array([0.37]),
            'hinge_cabinet': np.array([0., 1.45]),
            'microwave': np.array([-0.75]),
            'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
            }

        self.OBS_ELEMENT_BODY = {
            'bottom_burner': ["knob 2", "Burner 2"],
            'top_burner': ["knob 4", "Burner 4"],
            'light_switch': ["lightswitchroot", "lightblock_hinge"],
            'slide_cabinet': ["slidelink"],
            'hinge_cabinet': ["hingerightdoor"],
            'microwave': ["microdoorroot"],
            'kettle': ["kettleroot"],
        }

        self.model_site = ['end_effector', 'leftfinger', 'rightfinger', 'knob2_site','light_site', 'microhandle_site', 'kettle_site']

        self.OBS_ELEMENT_SITE = {
            'bottom_burner': 'knob2_site',
            'light_switch': 'light_site',
            'microwave': 'microhandle_site',
            'kettle': 'kettle_site',
        }

        self.BONUS_THRESH = 0.3

    def check_subtask_success(self, now_task, now_state):
        done_task = False
        element_idx = self.OBS_ELEMENT_INDICES[now_task]
        distance = np.linalg.norm(np.array(now_state)[element_idx] - self.OBS_ELEMENT_GOALS[now_task])
        complete = distance < self.BONUS_THRESH
        if complete:
            done_task = True
            print("environmet think {} is done".format(now_task))
        return done_task

class kitchen_franka_env(base_env):
    def __init__(self, input_config):
        super().__init__(input_config)
        self.task_params = TaskParameters()
        self.input_config = input_config
        self.start_arm_pose = input_config.start_arm_pose
        self.action_dim = input_config.action_dim
        self.obs_dim = input_config.obs_dim
        self.camera_names = input_config.camera_names
        self.robotic_joint_dims = input_config.robotic_joint_dims
        self.reset()

    def initialize_episode(self, physics):
        physics.named.data.qpos[:self.robotic_joint_dims] = self.start_arm_pose[:self.robotic_joint_dims]
        super().initialize_episode(physics)

    def before_step(self, action, physics):
        qpos = physics.data.qpos.copy()
        ctrl_feasible_position = np.zeros(action.shape[0])
        for j in range(self.action_dim):
            ctrl_feasible_position[j] = qpos[j] + action[j]*2*self.input_config.DT
        np.copyto(physics.data.ctrl, ctrl_feasible_position)

    def get_reward(self, physics):
        for element in self.episode_task:
            state_qpos = physics.data.qpos.copy()
            subtask_state = self.task_params.check_subtask_success(element, state_qpos)
            if subtask_state > 0:
                self.reward += 1
                self.episode_task.remove(element)
        return self.reward
    
    def reset(self):
        self.reward = 0
        self.episode_task = self.input_config.episode_task.copy()

class Kitchen_Grasp_Task():
    def __init__(self, input_config, show_camera_id = 0):
        self.reward = 0
        self.step_num = 0
        self.show_camera_id = show_camera_id
        self.input_config = input_config
        self.episode_task = self.input_config.episode_task.copy()
        self.kitchen_task = kitchen_franka_env(input_config)

    def check_episode_success(self):
        done_task = False
        if self.reward >= self.input_config.max_reward:
            done_task = True
        elif self.step_num >= self.input_config.max_episode_steps:
            done_task = True
        return done_task

    def reset(self):
        self.kitchen_task.reset()
        ts = self.env.reset()
        self.reward = 0
        self.step_num = 0
        self.episode_task = self.input_config.episode_task.copy()
        if self.input_config.show_episode:
            self.init_show_episode(ts)
        obs = ts.observation['env_state']
        reward = ts.reward
        done = self.check_episode_success()
        info = {}
        info['images'] = ts.observation['images'][self.input_config.camera_names[self.show_camera_id]]
        return obs, reward, done, info

    def step(self, action):
        ts = self.env.step(action)
        self.step_num += 1
        obs = ts.observation['env_state']
        reward = ts.reward
        done = self.check_episode_success()
        info = {}
        if self.input_config.show_episode:
            self.plt_img.set_data(ts.observation['images'][self.input_config.camera_names[self.show_camera_id]])
            plt.pause(0.02)
        info['images'] = ts.observation['images'][self.input_config.camera_names[self.show_camera_id]]
        return obs, reward, done, info
    
    def make_env(self):
        xml_path = self.input_config.xml_path
        physics = mujoco.Physics.from_xml_path(xml_path)
        control_timestep = self.input_config.DT
        time_limit  = self.input_config.time_limit
        self.env = control.Environment(physics, self.kitchen_task, time_limit=time_limit, control_timestep=control_timestep, n_sub_steps=None, flat_observation=False)
    
    def init_show_episode(self, ts):
        ax = plt.subplot()
        self.plt_img = ax.imshow(ts.observation['images'][self.input_config.camera_names[self.show_camera_id]])
        plt.ion()

def test_kitchen_sim():
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from utils.general_class import ParamDict
    from utils.helper import list_files_in_directory, read_csv_with_numpy_to_dict, read_dict_with_twodim_numpy_to_csv
    path_list = list_files_in_directory("/home/haowen/hw_RL_code/hw_kitchen/kitchen_panda_data/single_seq_data")
    # setup the environment
    xml_path = "/home/haowen/hw_RL_code/hw_kitchen//kitchen_panda/adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml"
    # xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')  
    # data0 = read_dict_with_twodim_numpy_to_csv(path_list)
    sreward = 0
    data0 = read_csv_with_numpy_to_dict(path_list[0])
    init_param = ParamDict()
    # init_param.start_arm_pose = data0[0]
    init_param.action_dim = 9
    init_param.robotic_joint_dims = 9
    init_param.obs_dim = 30
    init_param.camera_names = ['angle']
    init_param.max_reward = 4
    init_param.max_episode_steps = 1000000
    init_param.show_episode = True
    init_param.xml_path = xml_path
    init_param.DT = 0.08
    init_param.episode_task = ['bottom_burner', 'top_burner', 'light_switch', 'slide_cabinet', 'hinge_cabinet', 'microwave', 'kettle']
    # env = Kitchen_Grasp_Task(init_param)
    # env.make_env()
    # for id in range(len(path_list)):
    #     data = read_dict_with_twodim_numpy_to_csv(path_list[id])
    #     env.reset()
    #     for i in range(len(data['actions'])):
    #         action = data['actions'][i]
    #         obs, reward, done, info = env.step(action)
    #         print(reward)
    # path_list = "/home/haowen/hw_RL_code/hw_kitchen/kitchen_panda_data/single_task_data/begin to microwave.csv"
    # data = read_dict_with_twodim_numpy_to_csv(path_list)
    data = read_dict_with_twodim_numpy_to_csv("/home/haowen/hw_RL_code/act/data/microwave_kettle_bottom_burner_3.csv")
    actions = data['actions']
    states = data['states']
    init_param.start_arm_pose = states[0][0]
    for id in range(len(actions)):
        env = Kitchen_Grasp_Task(init_param)
        env.make_env()
        step_actions = []
        step_states = []
        action = actions[id]
        state = states[id]
        env.reset()
        for i in range(action.shape[0]):
            # actions = action[i]
            actions = np.random.random((9,)) +1
            obs, reward, done, info = env.step(actions)
            # sreward += reward
        print(reward)

if __name__ == '__main__':
    test_kitchen_sim()