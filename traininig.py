import gc
import torch
from ultralytics import YOLO

import run_configs as cfg
CONFIG_YAML = 'configs/config.yaml'

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def main():



    # model = YOLO('yolo11n.pt')
    # model.train(data=CONFIG_YAML, **cfg.YOLO11_NO_AUG)
    # clear_gpu_memory()
    #
    # model = YOLO('yolo26n.pt')
    # model.train(data=CONFIG_YAML, **cfg.YOLO26_NO_AUG)
    # clear_gpu_memory()
    #
    # model = YOLO('yolo11n.pt')
    # model.train(data=CONFIG_YAML, **cfg.YOLO11_WITH_AUG)
    # clear_gpu_memory()
    #
    # model = YOLO('yolo26n.pt')
    # model.train(data=CONFIG_YAML, **cfg.YOLO26_WITH_AUG)
    # clear_gpu_memory()

    model = YOLO('yolo26n.pt')
    model.train(data=CONFIG_YAML, **cfg.YOLO26_WITH_CUSTOM_AUG)
    clear_gpu_memory()

    model = YOLO('yolo11n.pt')
    model.train(data=CONFIG_YAML, **cfg.YOLO11_WITH_CUSTOM_AUG)
    clear_gpu_memory()

if __name__ == '__main__':
    main()