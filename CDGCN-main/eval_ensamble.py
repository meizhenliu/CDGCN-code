# import logging
# from lib import evaluation
# import os
#
# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# # save results
# os.system("test.py --dataset f30k --data_path /root/autodl-tmp/f30k --model_name ../checkpoint/f30k_best1 --save_results")
# os.system("test.py --dataset f30k --data_path /root/autodl-tmp/f30k --model_name ../checkpoint/f30k_best2 --save_results")
# # Evaluate model ensemble
# paths = ['../checkpoint/f30k_best1/results_f30k.npy',
#          '../checkpoint/f30k_best2/results_f30k.npy']
# print('-------------------------------------------------------------------------------------')
# #evaluation.eval_ensemble(results_paths=paths, fold5=True)
# evaluation.eval_ensemble(results_paths=paths, fold5=False)

# print('---------------------------------coco----------------------------------------------------')
# os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_best1 --save_results")
# os.system("python3 eval.py --dataset coco --data_path ../../data/coco --model_name runs/coco_best2  --save_results")
# # Evaluate model ensemble
# paths = ['runs/coco_best1/results_coco.npy',
#          'runs/coco_best2/results_coco.npy']
# print('-------------------------------------------------------------------------------------')
# evaluation.eval_ensemble(results_paths=paths, fold5=True)
# evaluation.eval_ensemble(results_paths=paths, fold5=False)
import os
import argparse
import logging
from lib import evaluation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# RUN_PATH = "../checkpoint/model_best.pth"
#
paths = ['../checkpoint/f30k_best1/results_coco.npy',
          '../checkpoint/f30k_best2/results_coco.npy']
evaluation.eval_ensemble(results_paths=paths, fold5=True)