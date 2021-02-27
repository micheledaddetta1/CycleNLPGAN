"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import logging
import time
import os

from tqdm import tqdm

from options.test_options import TestOptions
from models import create_model
import torch

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s: %(message)s')
    opt = TestOptions().parse()   # get training options

    torch.cuda.empty_cache()
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers


    #train_dataset, eval_dataset, test_dataset = create_dataset(opt, model)  # create a dataset given opt.dataset_mode and other options
    #dataset_size = len(train_dataset)    # get the number of images in the dataset.
    #logging.info('The number of training sentences = %d' % dataset_size)
    #logging.info('The number of evaluation sentences = %d' % len(eval_dataset))
    #logging.info('The number of test sentences = %d' % len(test_dataset))

    dataset_A = ["The pen is on the table", "I have teethache"]
    dataset_B = ["La penna è sul tavolo", "Ho mal di denti"]

    #G_A = torch.load("CycleNLPGAN/checkpoints/translation/latest_net_G_A.pth")
    model.set_input({"A" : dataset_A, "B" : dataset_B})  # unpack data from data loader
    model.evaluate()           # run inference
