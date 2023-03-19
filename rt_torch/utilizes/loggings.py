import logging
import os
from torch.utils.tensorboard import SummaryWriter
from rt_torch.utilizes.utilize import setup_seed


def log_init(args=None, rank=0):
    setup_seed(args.seed)
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        logfile = os.path.join(args.log_path, os.path.basename(args.log_path) + '.log')
        handler = logging.FileHandler(logfile, mode='a+')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.DEBUG)
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("Start print log")
        logger.info("seed: {}".format(args.seed))
        # filename_list = os.listdir('.')
        # expr = '\.py'
        # for filename in filename_list:
        #     if re.search(expr, filename) is not None:
        #         shutil.copyfile('./' + filename, os.path.join(log_path, filename))
        # with open(os.path.join(log_path, 'args.json'), 'w') as f:
        #     json.dump(args.__dict__, f)
        return logger, handler
    else:
        return None, None
    # return None, None, models_path, log_path, None

# def log_init(args=None, rank=0):
#     history = "/home/cz/bs/rt_torch/history"
#     # history = os.path.expanduser('~/history')
#     # log_name = time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '/'
#     if args.load_dir is not None:
#         log_name = os.path.basename(args.load_dir) + "_" + args.exp_name
#     else:
#         log_name = args.exp_name + "_" + args.text_encoder + "_" + str(args.lr_t) + "_" + str(args.lr_eff) + "_" + args.alias
#     log_path = os.path.join(history, log_name)
#     models_path = os.path.join(log_path, 'models')

#     # print(history)
#     # print(log_path)
#     # print(models_path)
#     os.makedirs(models_path, exist_ok=True)
#     setup_seed(args.seed)
#     # if rank == 0:
#     #     logger = logging.getLogger(__name__)
#     #     logger.setLevel(level=logging.INFO)
#     #     logfile = os.path.join(log_path, log_name + '.log')
#     #     handler = logging.FileHandler(logfile, mode='a+')
#     #     handler.setLevel(logging.DEBUG)
#     #     formatter = logging.Formatter('%(asctime)s - %(message)s')

#     #     # console_handler = logging.StreamHandler()
#     #     # console_handler.setLevel(logging.DEBUG)
#     #     # console_handler.setFormatter(formatter)
#     #     # logger.addHandler(console_handler)

#     #     handler.setFormatter(formatter)
#     #     logger.addHandler(handler)
#     #     logger.info("Start print log")
#     #     logger.info("seed: {}".format(args.seed))
#     #     writer = SummaryWriter(log_path)
#     #     # filename_list = os.listdir('.')
#     #     # expr = '\.py'
#     #     # for filename in filename_list:
#     #     #     if re.search(expr, filename) is not None:
#     #     #         shutil.copyfile('./' + filename, os.path.join(log_path, filename))
#     #     # with open(os.path.join(log_path, 'args.json'), 'w') as f:
#     #     #     json.dump(args.__dict__, f)
#     #     return logger, writer, models_path, log_path, handler
#     # else:
#     #     return None, None, models_path, log_path, None
#     return None, None, models_path, log_path, None
    
# # Load the arguments from a file
# with open('args.json', 'r') as f:
#     args_dict = json.load(f)

# # Create a new Namespace object from the saved arguments
# args = argparse.Namespace(**args_dict)