import logging

import numpy as np
import os
import sys
import ntpath
import time
from . import util
from subprocess import Popen, PIPE





class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False

        # create a logging file to store training losses
        self.log_names = [os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')]
        if self.opt.on_colab:
            self.log_names.append(os.path.join("/content/gdrive/My Drive/",opt.name,"loss_log.txt"))
        for log_name in self.log_names:
            with open(log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False


    # learning rate
    def print_current_lr(self, epoch, lr):
        """print current learning rate on console
        Parameters:
            epoch (int) -- current epoch
            lr (float) -- learning rate value
        """
        message = 'Learning rate at epoch %d : %.7f) ' % (epoch, lr)

        logging.info(message)  # print the message
        for log_name in self.log_names:
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        logging.info(message)  # print the message
        for log_name in self.log_names:
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message