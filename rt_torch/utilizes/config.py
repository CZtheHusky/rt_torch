from argparse import ArgumentParser


def str2bool(x):
    assert x == "True" or x == "False"
    return True if x == "True" else False


def get_parser_for_basic_args():
    parser = ArgumentParser("Basic Configuration")

    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--text_encoder', default="t5", type=str)
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--loader_shuffle', action='store_true', help="load the args")
    parser.add_argument('--loader_worker', default=16, type=int, help='')
    parser.add_argument('--test-iters', default=100, type=int, help='test_interval')
    parser.add_argument('--test-interval', default=5000, type=int, help='test_interval')
    parser.add_argument('--seed', default=100, type=int, help='')
    parser.add_argument('--save-interval', default=5000, type=int)
    parser.add_argument('--alias', default="", type=str, help="alias of the experiment")
    parser.add_argument('--sub_data', default="mix", type=str, help="data for training")
    parser.add_argument('--log-path', default="", type=str, help="")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_t', default=1e-4, type=float)
    parser.add_argument('--lr_eff', default=1e-5, type=float)
    parser.add_argument('--min-lr', default=1e-5, type=float)
    # parser.add_argument("--master-rank", default=0, type=int)
    parser.add_argument("--eval-eps", default=10, type=int)
    parser.add_argument("--eval-timeout", default=300, type=int)
    parser.add_argument('--load_args', action='store_true', help="load the args")
    parser.add_argument('--model', default="vanilla", type=str, help="")
    parser.add_argument('--quantile', default=True, type=str2bool, help="")


    parser.add_argument("--load-dir", type=str, help="Path of checkpoint to load.")

    parser.add_argument('--key_dim', default=512, type=int)
    parser.add_argument('--model_dim', default=512, type=int)
    parser.add_argument('--vocab_size', default=256, type=int)
    parser.add_argument('--num_actions', default=2, type=int)
    parser.add_argument('--token_learner_num', default=8, type=int)
    parser.add_argument('--seq_len', default=6, type=int)

    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Number of the Transformer encoders",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of attention heads for each attention layer in "
        "the Transformer decoder.",
    )
    parser.add_argument("--fp16", type=str2bool, default=True)
    return parser
