"""
Main function for training and evaluating MARL algorithms in NMARL envs
@author: Tianshu Chu
"""

"""
- This main is functional with all the configs, imports and methods for
IA2C, MA2C_NC, IA2C_CU, MA2C_CNET and MA2C_DIAL on RWare Small 4-Agent Scenario.
"""

# https://stackoverflow.com/questions/123198/how-to-copy-files

import numpy as np
import argparse
import configparser
import logging
import shutil
import threading
from torch.utils.tensorboard.writer import SummaryWriter
import gym
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
#from a1_cacc_env_ncomm import CACCEnv
from a2_customized_rware_env import CustomizeRware
from d_models_ncomm_rware import IA2C, IA2C_FP, MA2C_NC, IA2C_CU, MA2C_CNET, MA2C_DIAL
from e_utils_train_ncomm_rware import (Counter, Trainer, Tester, Evaluator, check_dir,
                                       copy_file, find_file, init_dir, init_log, init_test_flag)

# IA2C (MADDPG) -> RWare Small 4-Agent
# MA2C_NC (NeurComm) -> RWare Small 4-Agent
# IA2C_CU (ConsensusUpdate) -> RWare Small 4-Agent
# MA2C_CNET (CommNet) -> RWare Small 4-Agent
# MA2C_DIAL (DIAL) -> RWare Small 4-Agent

# Rename base dir and Recheck config dir and file names when run again
#default_base_dir = "1_IA2C_Rware_Small4Ag_eps_2e3"
#default_config_dir = './2_Configs_Small4Ag/config_1_ia2c_rwaresmall4ag_eps_2e3.ini'

#default_base_dir = "2_IA2C_CU_Rware_Small4Ag_eps_2e3"
#default_config_dir = './2_Configs_Small4Ag/config_2_ia2c_cu_rwaresmall4ag_eps_2e3.ini'

#default_base_dir = "3_MA2C_NC_Rware_Small4Ag_eps_2e3"
#default_config_dir = './2_Configs_Small4Ag/config_3_ma2c_nc_rwaresmall4ag_eps_2e3.ini'

#default_base_dir = "4_MA2C_CNET_Rware_Small4Ag_eps_2e3"
#default_config_dir = './2_Configs_Small4Ag/config_4_ma2c_cnet_rwaresmall4ag_eps_2e3.ini'

#default_base_dir = "5_MA2C_DIAL_Rware_Small4Ag_eps_2e3"
#default_config_dir = './2_Configs_Small4Ag/config_5_ma2c_dial_rwaresmall4ag_eps_2e3.ini'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    args = parser.parse_args()
    return args

def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    #copy_file(config_dir, dirs['data'])  # Originally
    shutil.copy2(config_dir, dirs['data'])  # By me
    config = configparser.ConfigParser()
    config.read(config_dir)

    # Initialize step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    #model = init_agent(env, config['MODEL_CONFIG'], total_step, seed)  # Originally

    # Initialize env
    env = CustomizeRware(config.get('ENV_CONFIG', 'env_scenario'),
                         config.get('ENV_CONFIG', 'scenario'),
                         config.get('ENV_CONFIG', 'ma_algo_name'))
    ma_algo_name = config.get('ENV_CONFIG', 'ma_algo_name')

    # Get obs and action dims
    observation_sizes_ma2c, action_sizes_ma2c = (env.initialize_env_dims_ma2c())
    print("Observation sizes for ma2c: ", observation_sizes_ma2c,
          "\nAction sizes for ma2c: ", action_sizes_ma2c)
    logging.info('Training: Action dims %r, Agent dims: %d' %
                 (action_sizes_ma2c, env.n_agents))

    """
    # Disable this condition for now
    if ma_algo_name.startswith('ia2c'):
        observation_sizes_ia2c, action_sizes_ia2c = (env.initialize_env_dims_ia2c())
        print("Observation sizes for ia2c: ", observation_sizes_ia2c,
              "\nAction sizes for ia2c: ", action_sizes_ia2c)
        logging.info('Training: Action dims %r, Agent dims: %d' %
                     (action_sizes_ia2c, env.n_agents))
        model = IA2C(observation_sizes_ia2c, action_sizes_ia2c, env.neighbor_mask, env.distance_mask,
                     config.getint('ENV_CONFIG', 'coop_gamma'), total_step,
                     config['MODEL_CONFIG'], seed=seed)

    else:
        observation_sizes_ma2c, action_sizes_ma2c = (env.initialize_env_dims_ma2c())
        print("Observation sizes for ma2c: ", observation_sizes_ma2c,
              "\nAction sizes for ma2c: ", action_sizes_ma2c)
        logging.info('Training: Action dims %r, Agent dims: %d' %
                     (action_sizes_ma2c, env.n_agents))
        model = MA2C_NC(observation_sizes_ma2c, action_sizes_ma2c, env.neighbor_mask, env.distance_mask,
                        config.getint('ENV_CONFIG', 'coop_gamma'), total_step,
                        config['MODEL_CONFIG'], seed=seed)
    """

    if ma_algo_name == 'ia2c':
        model = IA2C(observation_sizes_ma2c, action_sizes_ma2c, env.neighbor_mask, env.distance_mask,
                     config.getint('ENV_CONFIG', 'coop_gamma'), total_step, config['MODEL_CONFIG'],
                     seed=seed)
    elif ma_algo_name == 'ma2c_cu':
        model = IA2C_CU(observation_sizes_ma2c, action_sizes_ma2c, env.neighbor_mask, env.distance_mask,
                        config.getint('ENV_CONFIG', 'coop_gamma'), total_step, config['MODEL_CONFIG'],
                        seed=seed)
    elif ma_algo_name == 'ma2c_nc':
        model = MA2C_NC(observation_sizes_ma2c, action_sizes_ma2c, env.neighbor_mask, env.distance_mask,
                        config.getint('ENV_CONFIG', 'coop_gamma'), total_step, config['MODEL_CONFIG'],
                        seed=seed)
    elif ma_algo_name == 'ma2c_cnet':
        # This is actually CommNet
        model = MA2C_CNET(observation_sizes_ma2c, action_sizes_ma2c, env.neighbor_mask, env.distance_mask,
                          config.getint('ENV_CONFIG', 'coop_gamma'), total_step, config['MODEL_CONFIG'],
                          seed=seed)
    elif ma_algo_name == 'ma2c_dial':
        model = MA2C_DIAL(observation_sizes_ma2c, action_sizes_ma2c, env.neighbor_mask, env.distance_mask,
                          config.getint('ENV_CONFIG', 'coop_gamma'), total_step, config['MODEL_CONFIG'],
                          seed=seed)
    else:
        print()
        print("No Name for Algorithm is passed to interact with the Environment..")
        return None

    model.load(dirs['model'], train_mode=True)

    # Disable multi-threading for safe SUMO implementation
    summary_writer = SummaryWriter(dirs['log'], flush_secs=10000)
    trainer = Trainer(env, model, global_counter, summary_writer, config["TRAIN_CONFIG"], config["ENV_CONFIG"],
                      output_path=dirs['data'])
    trainer.run()

    # Save model
    final_step = global_counter.cur_step
    model.save(dirs['model'], final_step)
    summary_writer.close()

def evaluate_fn(agent_dir, output_dir, seeds, port, demo):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file
    config_dir = find_file(agent_dir + '/data/')
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'], port=port)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0)
    if model is None:
        return
    model_dir = agent_dir + '/model/'
    if not model.load(model_dir):
        return
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir, gui=demo)
    evaluator.run()

def evaluate(args):
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        output_dir = None
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds, 1, args.demo)

if __name__ == '__main__':
    args = parse_args()
    main = train(args)  # By me

    """
    # Originally
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
    """
