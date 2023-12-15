import os
import time
from mindspore import save_checkpoint, context
from mindspore.train.callback import Callback
from src.utils.local_adapter import get_rank_id

class CallbackSaveByIoU(Callback):
    """SaveCallback"""
    def __init__(self, eval_model, ds_eval, eval_period=1, eval_start=1, save_path=None):
        """init"""
        super(CallbackSaveByIoU, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.mIoU = 0.
        self.mIoU_img = 0.
        self.eval_period = eval_period
        self.save_path = save_path
        self.eval_start = eval_start

    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        rank_id = get_rank_id()
        if ((cur_epoch + 1) % self.eval_period) == 0:
            if cur_epoch < self.eval_start:
                return
            print("Start evaluate...")
            result = self.model.eval(self.ds_eval, dataset_sink_mode=False)
            mIoU = result["mIoU"]
            if mIoU > self.mIoU:
                self.mIoU = mIoU
                file_name = f"best_model_dev_{rank_id}.ckpt"
                save_path = os.path.join(self.save_path, file_name)
                print("Save model...")
                save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=save_path)
            print(f"Device:{rank_id}, Epoch:{cur_epoch}, mIoU:{mIoU:.5f}")

    def end(self, run_context):
        _ = run_context.original_args()
        rank_id = get_rank_id()
        print(f"Device:{rank_id}, Best mIoU:{(self.mIoU*100):.2f}%")


class RecorderCallback(Callback):
    """Callback base class"""
    def __init__(self, recorder):
        self.recorder = recorder

    def step_begin(self, run_context):
        """Called before each step beginning."""
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        cur_time = time.time()
        cur_step = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)
        if context.get_context('device_target') == 'Ascend':
            log_str = ">>> {} Epoch[{:03d}/{:03d}] Step[{:04d}/{:04d}] Loss[{:.3f}] Time[{:.3f}s] SysTime[{}] ".format(
                "Train", int(cb_params.cur_epoch_num), int(cb_params.epoch_num),
                cur_step,
                int(cb_params.batch_num), float(cb_params.net_outputs[0]), cur_time - cb_params.init_time,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        else:
            log_str = ">>> {} Epoch[{:03d}/{:03d}] Step[{:04d}/{:04d}] Loss[{:.3f}] Time[{:.3f}s] SysTime[{}] ".format(
                "Train", int(cb_params.cur_epoch_num), int(cb_params.epoch_num),
                cur_step,
                int(cb_params.batch_num), float(cb_params.net_outputs), cur_time - cb_params.init_time,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if cur_step % 100 == 0:
            if context.get_context('device_target') == 'GPU':
                if self.recorder is not None:
                    self.recorder.logger.info(log_str)
            else:
                print(log_str)

    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params.epoch_init_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_time = time.time()
        if context.get_context('device_target') == 'Ascend':
            log_str = ">>> {} Epoch[{:03d}/{:03d}] Loss[{:.3f}] Time[{}s]".format(
                "Train", int(cb_params.cur_epoch_num), int(cb_params.epoch_num),
                float(cb_params.net_outputs[0]),
                time.strftime('%H:%M:%S', time.gmtime(cur_time - cb_params.epoch_init_time)))
            print(log_str)
        else:
            log_str = ">>> {} Epoch[{:03d}/{:03d}] Loss[{:.3f}] Time[{}s]".format(
                "Train", int(cb_params.cur_epoch_num), int(cb_params.epoch_num),
                float(cb_params.net_outputs),
                time.strftime('%H:%M:%S', time.gmtime(cur_time - cb_params.epoch_init_time)))
            if self.recorder is not None:
                self.recorder.logger.info(log_str)

