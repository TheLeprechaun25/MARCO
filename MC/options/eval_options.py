import argparse


def get_options():
    parser = argparse.ArgumentParser(description="Marco")

    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    # Environment
    parser.add_argument('--n', type=int, default=20, help='Problem size (only for training)')

    # Memory
    parser.add_argument('--memory_type', type=str, default='shared', choices=['shared', 'individual', 'op_based', 'none'], help='Memory type: shared, individual, operation-based, none')
    parser.add_argument('--memory_size', type=int, default=100000, help='Memory size')
    parser.add_argument('--k', type=int, default=20, help='k')
    parser.add_argument('--mem_aggr', type=str, default='linear', choices=['sum', 'linear', 'exp'], help='The aggregation method for the memory. Sum of all k values. Linear weighted sum. Exponential weighted sum.')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim')
    parser.add_argument('--n_layers', type=int, default=3, help='n_layers')
    parser.add_argument('--shared_weights', action='store_true', help='Use shared weights')
    parser.add_argument('--n_heads', type=int, default=8, help='n_heads')
    parser.add_argument('--normalization', type=str, default='layer', choices=['batch', 'instance', 'layer', 'rms'], help='Normalization')
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'swiglu', 'relu'], help='Activation')
    parser.add_argument('--bias', action='store_true', help='Use bias')
    parser.add_argument('--tanh_clipping', type=float, default=10, help='Tanh clipping')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision. Only for evaluation')

    # Eval
    parser.add_argument('--eval_graph_type', type=str, default='er200', help='Type of graphs to be used (rb200300, rb8001200, er700800). For rand ER instances use er{n}')
    parser.add_argument('--compile', action='store_true', help='Compile pytorch model')
    parser.add_argument('--save_all_time', action='store_true', help='Save all time')
    parser.add_argument('--n_solutions', type=int, default=50, help='Number of threads')
    parser.add_argument('--topk', type=int, default=1, help='Executed actions in each inference step. topk=1 for default inference')
    parser.add_argument('--max_steps', type=int, default=2, help='Max test steps. 2 refers to 2*n.')
    parser.add_argument('--patience', type=int, default=1e9, help='Patience')
    parser.add_argument('--n_graphs', type=int, default=3, help='Number of graphs')

    # Others
    parser.add_argument('--model_load_path', type=str, default='used_models/marco_checkpoint.pth', help='Model load path')
    parser.add_argument('--save_results', type=bool, default=False, help='Save results')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose')

    args = parser.parse_args()
    # group all arguments in different dictionaries
    env_params = {
        'problem_size': args.n,
    }
    memory_params = {
        'memory_type': args.memory_type,
        'memory_size': args.memory_size,
        'k': args.k,
        'mem_aggr': args.mem_aggr,
    }
    model_params = {
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'shared_weights': args.shared_weights,
        'n_heads': args.n_heads,
        'normalization': args.normalization,
        'activation': args.activation,
        'bias': args.bias,
        'tanh_clipping': args.tanh_clipping,
        'dropout': args.dropout,
        'mixed_precision': args.mixed_precision,
    }
    eval_params = {
        'compile': args.compile,
        'save_all_time': args.save_all_time,  # save all time or only the last time step
        'seed': args.seed,
        'n_solutions': args.n_solutions,
        'topk': args.topk,
        'max_steps': args.max_steps,
        'patience': args.patience,
        'eval_graph_type': args.eval_graph_type,
        'n_graphs': args.n_graphs,

        # Main directory
        'model_load_path': args.model_load_path,  # directory path of pre-trained model and log files saved.
        'save_results': args.save_results,
        'verbose': args.verbose,
    }

    return env_params, memory_params, model_params, eval_params







