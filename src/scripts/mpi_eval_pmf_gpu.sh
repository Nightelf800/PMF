#!/usr/bin/env bash
mpirun -n 2 python eval.py --config_path src/config/eval_server_kitti.yaml --device_target GPU

