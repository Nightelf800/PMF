import os
import shutil
import yaml
from src.utils.utils import is_main_process


class Option(object):
    def __init__(self, args):
        self.config_path = args.config_path
        self.config = yaml.safe_load(open(args.config_path, "r"))
        self.device_target = args.device_target

        # ---------------------------- general options -----------------
        self.save_path = self.config["save_path"]  # log path
        self.seed = self.config["seed"]  # manually set RNG seed
        self.gpu = self.config["gpu"]  # GPU id to use, e.g. "0,1,2,3"
        self.rank = 0  # rank of distributed thread
        self.world_size = 1
        self.distributed = False
        self.n_gpus = len(self.gpu.split(","))  # # number of GPUs to use by default
        self.dist_backend = "nccl"
        self.dist_url = "env://"

        self.print_frequency = self.config["print_frequency"]  # print frequency (default: 10)
        self.n_threads = self.config["n_threads"]  # number of threads used for data loading
        self.experiment_id = self.config["experiment_id"]  # identifier for experiment
        self.is_debug = self.config["is_debug"]
        # --------------------------- data config ------------------------
        self.dataset = self.config["dataset"]
        self.nclasses = self.config["nclasses"]
        self.data_root = self.config["data_root"]
        self.has_label = self.config["has_label"]
        self.batch_size = self.config["batch_size"]

        # --------------------------- model options -----------------------
        self.base_channels = self.config["base_channels"]
        self.img_backbone = self.config["img_backbone"]
        self.imagenet_pretrained = self.config["imagenet_pretrained"]
        self.pretrained_path = self.config["pretrained_path"]

        # --------------------------- checkpoit model ----------------------
        self.best_model_path = self.config["best_model_path"]
        self.lambda_ = self.config["lambda"]
        self.gamma = self.config["gamma"]
        self.tau = self.config["tau"]

        # self._prepare()

    def _prepare(self):
        # ---- check params
        # --------------------------------
        if not os.path.isdir(self.save_path):
            raise ValueError("pretrained model is required, please train your model first. Path not exist: {}"
                             .format(self.save_path))

    def check_path(self):
        if is_main_process:
            if os.path.exists(self.save_path):
                print("file exist: {}".format(self.save_path))
                action = input("Select Action: d(delete) / q(quit): ").lower().strip()
                if action == "d":
                    shutil.rmtree(self.save_path)
                else:
                    raise OSError("Directory exits: {}".format(self.save_path))

            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
