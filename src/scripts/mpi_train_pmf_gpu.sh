#!/usr/bin/env bash
mpirun -n 2 python train.py --config_path src/config/train_server_kitti.yaml --device_target GPU

