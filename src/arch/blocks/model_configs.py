import os
from basicts.utils import get_regular_settings, load_adj
from baselines.STID.arch.stid_arch import STID
from baselines.AGCRN.arch.agcrn_arch import AGCRN

MODEL_CONFIGS ={
    'STID': STID,
    'AGCRN': AGCRN
}

def get_model_config(MODEL_NAME, DATA_NAME):
    regular_settings = get_regular_settings(DATA_NAME)
    INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
    OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
    config = {}
    if MODEL_NAME == 'STID':
        config = {
            'PEMS03': {
                "num_nodes": 358,
                "input_len": INPUT_LEN,
                "input_dim": 3,
                "embed_dim": 32,
                "output_len": OUTPUT_LEN,
                "num_layer": 3,
                "if_node": True,
                "node_dim": 32,
                "if_T_i_D": True,
                "if_D_i_W": True,
                "temp_dim_tid": 32,
                "temp_dim_diw": 32,
                "time_of_day_size": 288,
                "day_of_week_size": 7
            },
            'PEMS04': {
                    "num_nodes": 307,
                    "input_len": INPUT_LEN,
                    "input_dim": 3,
                    "embed_dim": 32,
                    "output_len": OUTPUT_LEN,
                    "num_layer": 3,
                    "if_node": True,
                    "node_dim": 32,
                    "if_T_i_D": True,
                    "if_D_i_W": True,
                    "temp_dim_tid": 32,
                    "temp_dim_diw": 32,
                    "time_of_day_size": 288,
                    "day_of_week_size": 7
            },
            'PEMS07':{
                    "num_nodes": 883,
                    "input_len": INPUT_LEN,
                    "input_dim": 3,
                    "embed_dim": 32,
                    "output_len": OUTPUT_LEN,
                    "num_layer": 3,
                    "if_node": True,
                    "node_dim": 32,
                    "if_T_i_D": True,
                    "if_D_i_W": True,
                    "temp_dim_tid": 32,
                    "temp_dim_diw": 32,
                    "time_of_day_size": 288,
                    "day_of_week_size": 7
            },
            'PEMS08':{
                    "num_nodes": 170,
                    "input_len": INPUT_LEN,
                    "input_dim": 3,
                    "embed_dim": 32,
                    "output_len": OUTPUT_LEN,
                    "num_layer": 3,
                    "if_node": True,
                    "node_dim": 32,
                    "if_T_i_D": True,
                    "if_D_i_W": True,
                    "temp_dim_tid": 32,
                    "temp_dim_diw": 32,
                    "time_of_day_size": 288,
                    "day_of_week_size": 7
            }
        }
    else:
        raise ValueError(f"Model {MODEL_NAME} not found in configurations. Available: {list(MODEL_CONFIGS.keys())}")
    return config[DATA_NAME]

def infer_dataset_from_path(checkpoint_path):
    path_parts = checkpoint_path.split(os.sep)
    datasets = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'SD', 'GLA', 'GBA']
    for part in path_parts:
        for dataset in datasets:
            if dataset in part:
                return dataset
    return None 