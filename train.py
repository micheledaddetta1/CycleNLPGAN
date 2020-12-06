"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import logging
import time

from tqdm import tqdm

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s: %(message)s')
    opt = TrainOptions().parse()   # get training options

    torch.cuda.empty_cache()
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers


    train_dataset, eval_dataset, test_dataset = create_dataset(opt, model)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    logging.info('The number of training sentences = %d' % dataset_size)
    logging.info('The number of evaluation sentences = %d' % len(eval_dataset))
    logging.info('The number of test sentences = %d' % len(test_dataset))

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = opt.iter_count                # the total number of training iterations


    n = round(opt.iter_count/opt.batch_size) #NBatch totali
    n -= (opt.epoch_count-1)*(round(len(train_dataset)/opt.batch_size))

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            epoch_iter += opt.batch_size

            if epoch == opt.epoch_count:
                if n > 0:
                    n -= 1
                    continue
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)


            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logging.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        logging.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

        sentences_filename = "eval_sentences.txt"
        distance_filename = "distances.txt"
        top_k_filename = "top_k.txt"
        with open(distance_filename, "a") as distance_file:
            distance_file.write("NEW EPOCH:\n")
        with open(top_k_filename, "a") as top_file:
            top_file.write("NEW EPOCH:\n")
        with open(sentences_filename, "a") as sentences_file:
            sentences_file.write("NEW EPOCH:\n")

        for i, data in enumerate(eval_dataset):  # inner loop within one epoch
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.evaluate(sentences_file=sentences_filename, distance_file=distance_filename, top_k_file=top_k_filename)

        with open(distance_filename, "a") as distance_file:
            distance_file.write("\n\n\n\n")
        with open(top_k_filename, "a") as top_file:
            top_file.write("\n\n\n\n")
        with open(sentences_filename, "a") as sentences_file:
            sentences_file.write("\n\n\n\n")

        logging.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        torch.cuda.empty_cache()
