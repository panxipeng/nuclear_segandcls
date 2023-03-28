import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "smile"
        model_mode = "fast" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = 5 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        # aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        # act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
        # out_shape = [80, 80] # patch shape at output of network

        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [256, 256] # patch shape at output of network

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [256,256]:
                raise Exception("If using `original` mode, input shape must be [256,256] and output shape must be [164,164]")

        # if model_mode == "fast":
        #     if act_shape != [256,256] or out_shape != [164,164]:
        #         raise Exception("If using `original` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "monusac" # extracts dataset info from dataset.py

        self.log_dir = "/DataDisk1/cjj/workspace/mia2022/mia2023/" # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [
        "/DataDisk2/cjj/workspace/SMLILE/code+data/dataset/MoNuSAC/MoNuSAC_processed_trainingData/training_data_20220707/monusac/monusac/train/540x540_80x80"
            # "/mnt/disk4T_TS/FromJuchiyun/workspace/hover_net-CCA_v0/dataset/training_data_fold2_0706/monusac/monusac/train/540x540_80x80"
        ]

        self.valid_dir_list = [
            "/DataDisk2/cjj/workspace/SMLILE/code+data/dataset/MoNuSAC/MoNuSAC_processed_trainingData/training_data_20220707/monusac/monusac/valid/540x540_80x80"
            # "/mnt/disk4T_TS/FromJuchiyun/workspace/hover_net-CCA_v0/dataset/training_data_fold2_0706/monusac/monusac/valid/540x540_80x80"
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
