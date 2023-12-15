import argparse
import os
import time
import mindspore
import mindspore.dataset as ds
from mindspore import context, ParallelMode, load_checkpoint, load_param_into_net, Model, TimeMonitor
from mindspore.communication import init, get_rank, get_group_size
from src.config.eval_option import Option
from src.dataset.perspective_view_loader import PerspectiveViewLoader
from src.dataset.semantic_kitti.parser import SemanticKitti
from src.models.pmf_net import PMFNet
from src.utils.common import CustomWithLossCell, CustomMutiLoss
from src.utils.metric import IOUEval, CustomWithEvalCell

# 数据集解压路径，仅限Ascend，如有需要自行修改
dataset_name = "SemanticKitti"

parser = argparse.ArgumentParser(description='MindSpore PMF')
parser.add_argument('--multi_data_url',
                    help='使用单数据集或多数据集时，需要定义的参数',
                    default='[{}]')

parser.add_argument('--pretrain_url',
                help='非必选，只有在界面上选择模型时才需要，使用单模型或多模型时，需要定义的参数',
                default='[{}]')
parser.add_argument('--train_url',
                    help='必选，回传结果到启智，需要定义的参数',
                    default='')
parser.add_argument("--config_path", type=str, metavar="config_path",
                    help="path of config file, type: string")
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'GPU'],
    help='device where the code will be implemented (default: Ascend)')


def evaluation(settings):
    settings = settings
    rank = int(os.getenv('RANK_ID', '0'))
    mindspore.set_seed(settings.seed)

    print("-->训练使用设备: ", settings.device_target)
    if settings.device_target == "Ascend":
        device_num = int(os.getenv('RANK_SIZE'))
        rank = int(os.getenv('RANK_ID'))

        if device_num == 1:
            ###拷贝数据集到训练环境
            context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
            DatasetToEnv(args.multi_data_url, data_dir)
            pretrain_to_env(args.pretrain_url, pretrain_dir)
            is_distributed = False
        else:
            # set device_id and init for multi-card training
            context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target,
                                device_id=int(os.getenv('ASCEND_DEVICE_ID')))
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            # Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
            if rank % 8 == 0:
                ###拷贝数据集到训练环境
                DatasetToEnv(args.multi_data_url, data_dir)
                pretrain_to_env(args.pretrain_url, pretrain_dir)
                # Set a cache file to determine whether the data has been copied to obs.
                # If this file exists during multi-card training, there is no need to copy the dataset multiple times.
                f = open("/cache/download_input.txt", 'w')
                f.close()
                try:
                    if os.path.exists("/cache/download_input.txt"):
                        print("download_input succeed")
                except Exception:
                    print("download_input failed")
            while not os.path.exists("/cache/download_input.txt"):
                time.sleep(1)
            is_distributed = True

        # 数据集路径,需要根据实际需求修改
        data_root = os.path.join(data_dir, dataset_name, "SemanticKitti/dataset/sequences")
        # 预训练模型路径,需要根据实际需求修改
        data_config_path = "/cache/user-job-dir/code/src/dataset/semantic_kitti/semantic-kitti.yaml"

        recorder = None

        if is_distributed:
            best_model_path = os.path.join(pretrain_dir, f"best_model_dev_{rank}.ckpt")
            param_dict = load_checkpoint(best_model_path)
        else:
            best_model_path = os.path.join(pretrain_dir, f"best_model_dev_{rank}.ckpt")
            param_dict = load_checkpoint(best_model_path)

    else:
        # 分布式运行or单卡运行
        print("-->GPU数量: ", settings.n_gpus)
        rank = int(os.getenv('RANK_ID', '0'))
        if settings.n_gpus > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu
            context.set_context(mode=context.PYNATIVE_MODE, device_target=settings.device_target)
            init()
            rank = get_rank()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            is_distributed = True
            if rank == 0:
                recorder = Recorder(settings, settings.save_path)
            else:
                recorder = 0
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target=settings.device_target,
                                device_id=int(settings.gpu))
            is_distributed = False
            recorder = Recorder(settings, settings.save_path)

        data_root = settings.data_root
        data_config_path = "src/dataset/semantic_kitti/semantic-kitti.yaml"

        if is_distributed:
            best_model_path = os.path.join(settings.best_model_path, f"best_model_dev_{rank}.ckpt")
            param_dict = load_checkpoint(best_model_path)
        else:
            best_model_path = os.path.join(settings.best_model_path, f"best_model_dev_{rank}.ckpt")
            param_dict = load_checkpoint(best_model_path)


    print("------------------加载模型和训练权重----------------")
    # model init
    net = PMFNet(
        pcd_channels=5,
        img_channels=3,
        nclasses=settings.nclasses,
        base_channels=settings.base_channels,
        image_backbone=settings.img_backbone,
        imagenet_pretrained=False
    )

    # load checkpoint
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # data init
    valset = SemanticKitti(
        root=data_root,
        sequences=[8],
        config_path=data_config_path
    )
    cls_weight = 1 / (valset.cls_freq + 1e-3)
    ignore_class = []
    for cl, v in enumerate(cls_weight):
        if valset.data_config["learning_ignore"][cl]:
            cls_weight[cl] = 0
        if cls_weight[cl] < 1e-10:
            ignore_class.append(cl)
    if recorder is not None:
        recorder.logger.info("weight: {}", cls_weight)
    val_pv_loader = PerspectiveViewLoader(
        dataset=valset,
        config=settings.config,
        is_train=False)
    if is_distributed:
        rank_size = get_group_size()
        val_loader = ds.GeneratorDataset(
            val_pv_loader,
            column_names=["pcd", "img", "mask", "label"],
            shuffle=False,
            shard_id=rank,
            num_shards=rank_size)
    else:
        val_loader = ds.GeneratorDataset(
            val_pv_loader,
            column_names=["pcd", "img", "mask", "label"],
            num_parallel_workers=settings.n_threads,
            max_rowsize=32,
            shuffle=False)
    val_loader = val_loader.batch(
        batch_size=settings.batch_size,
        num_parallel_workers=settings.n_threads,
        drop_remainder=False)

    # metric init
    loss = CustomMutiLoss(settings, cls_weight)
    loss_net = CustomWithLossCell(settings, net, loss)
    eval_net = CustomWithEvalCell(net)
    metric = {"mIoU": IOUEval(settings.nclasses,
                              recorder=recorder,
                              ignore=ignore_class,
                              is_distributed=is_distributed)}
    model = Model(loss_net, eval_network=eval_net, metrics=metric)
    time_cb = TimeMonitor(val_loader.get_dataset_size())
    print("------------------执行推理----------------")
    result = model.eval(val_loader, dataset_sink_mode=False, callbacks=[time_cb])
    mIoU = result["mIoU"]
    print("------------------执行结果----------------")
    print(f"Best mIoU:{(mIoU * 100):.2f}%")




if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    if args.device_target == "Ascend":
        from openi import openi_multidataset_to_env as DatasetToEnv, pretrain_to_env
        # main.main(['install', '-r', '/cache/code/pmf/requirements.txt'])
        data_dir = '/cache/data'
        result_dir = '/cache/result'
        pretrain_dir = '/cache/pretrainmodel'

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)
    elif args.device_target == "GPU":
        from src.utils.recorder import Recorder
    else:
        raise ValueError("Unsupported platform.")

    evaluation(Option(args))
