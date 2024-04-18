import logging
from utils import create_logger
from TSPTester import TSPTester as Tester


USE_CUDA = True


env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'normalize': False,
    'memory_type': 'shared',  # none, individual, shared
    'mem_aggr': 'linear',  # linear, sum
    'k': 10, # number of neighbors to retrieve from memory

    'retrieve_freq': 10,  # retrieve from memory every n steps
}

model_params = {
    'model_size': 'large',  # 'small', 'medium', 'large
    'dropout': 0.0,
    'bias': False,
    'activation': 'swiglu',
    'norm': 'layer',
    'logit_clipping': 10,
    'eval_type': 'argmax',
    'use_memory': env_params['memory_type'] != 'none',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'compile': False,
    'model_load': {
        'path': './result/tsp_cl100200',  # directory path of pre-trained model and log files saved.
        'epoch': 180,  # epoch version of pre-trained model to load.
    },

    'n_samples': 10,
    'eval_sizes': [500],
    'eval_n_instances': [10],
    'pomo_sizes': [50],
    'aug_factor': 1,
}

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20',
        'filename': 'run_log'
    }
}


def main():
    # Model size
    if model_params['model_size'] == 'small':
        model_params['embedding_dim'] = 128
        model_params['encoder_layer_num'] = 6
        model_params['head_num'] = 8
        model_params['ff_hidden_dim'] = 512

    elif model_params['model_size'] == 'medium':
        model_params['embedding_dim'] = 256
        model_params['encoder_layer_num'] = 6
        model_params['head_num'] = 8
        model_params['ff_hidden_dim'] = 1024

    elif model_params['model_size'] == 'large':
        model_params['embedding_dim'] = 512
        model_params['encoder_layer_num'] = 6
        model_params['head_num'] = 16
        model_params['ff_hidden_dim'] = 2048

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    tester.run()


def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()
