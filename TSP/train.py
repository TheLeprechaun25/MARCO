import logging
from utils import create_logger
from TSPTrainer import TSPTrainer as Trainer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--problem_size', type=int, default=20)
parser.add_argument('--model_size', type=str, default='large', choices=['small', 'medium', 'large'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--norm', type=str, default='layer', choices=['layer', 'rms', 'instance'])
parser.add_argument('--activation', type=str, default='swiglu', choices=['gelu', 'relu', 'swiglu'])
parser.add_argument('--epochs', type=int, default=250)

parser.add_argument('--k', type=int, default=10)
parser.add_argument('--punish_type', type=str, default='avg', choices=['avg', 'max'])
parser.add_argument('--punishment', type=float, default=0.02)


args = parser.parse_args()

env_params = {
    'problem_size': args.problem_size,
    'pomo_size': min(args.problem_size, 100),
    'normalize': True,
    'memory_type': 'individual',  # none, individual, shared
    'mem_aggr': 'linear',  # sum, linear
    'k': args.k,  # number of neighbors to retrieve from memory
    'repeat_punishment': args.punishment,  # Punishment for repeating edges or similarity with kNN solutions
    'punish_type': args.punish_type,  # avg, max   (take the average similarity or the max similarity)
    'retrieve_freq': 1,  # retrieve from memory every n steps
}

model_params = {
    'model_size': args.model_size,  # 'small', 'medium', 'large'
    'dropout': 0.0,
    'bias': False,
    'activation': args.activation,  # 'gelu', 'relu', 'swiglu'
    'norm': args.norm,  # 'layer', 'rms'
    'logit_clipping': 10,
    'eval_type': 'argmax',
    'use_memory': env_params['memory_type'] != 'none',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6,
    },
    'scheduler': {
        'milestones': [50, 100, 150, 200],
        'gamma': 0.9
    }
}

trainer_params = {
    'use_cuda': True,
    'compile': False,
    'rand_train_sizes': False,
    'finetune': True,  # Only train decoder
    'epochs': args.epochs,
    'train_episodes': 1 * args.batch_size,
    'train_batch_size': args.batch_size,
    'max_grad_norm': 1.0,
    'model_save_interval': 1,
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': f'result/tsp_cl100200',  # directory path of pre-trained model and log files saved.
        'epoch': 180,  # epoch version of pre-trained model to load.
    },
    'test_sizes': [200],
    'test_batch_sizes': [10],
    'test_pomo_sizes': [20],
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n20',
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

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    trainer.run()


def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()
