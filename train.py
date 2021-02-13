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
import os
import gc
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


    #train_dataset, eval_dataset, test_dataset = create_dataset(opt, model)  # create a dataset given opt.dataset_mode and other options
    train_dataset, eval_dataset = create_dataset(opt, model)
    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    logging.info('The number of training sentences = %d' % dataset_size)
    logging.info('The number of evaluation sentences = %d' % len(eval_dataset))
   # logging.info('The number of test sentences = %d' % len(test_dataset))
    logging.info('The number of training batches = %d' % len(train_dataset.dataloader))
    logging.info('The number of evaluation batches = %d' % len(eval_dataset.dataloader))
    #logging.info('The number of test batches = %d' % len(test_dataset.dataloader))

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = opt.iter_count                # the total number of training iterations


    n = round(opt.iter_count/opt.batch_size) #NBatch totali
    n -= (opt.epoch_count-1)*(round(len(train_dataset)/opt.batch_size))
    previous_suffix = None



    if not opt.continue_train:

        fw = open("0_0_distance.txt", "w")
        fw.close()
        fw = open("0_0_sentence.txt", "w")
        fw.close()
        fw = open("0_0_top_k.txt", "w")
        fw.close()
        fw = open("0_0_mutual_distance_A.txt", "w")
        fw.close()
        fw = open("0_0_mutual_distance_B.txt", "w")
        fw.close()

        for j, eval_data in enumerate(eval_dataset.dataloader):  # inner loop within one epoch
            if j > 20:
                break
            model.set_input(eval_data)  # unpack data from dataset and apply preprocessing
            model.evaluate(sentences_file="0_0_sentence.txt", distance_file="0_0_distance.txt", mutual_avg_file_A="0_0_mutual_distance_A.txt", mutual_avg_file_B="0_0_mutual_distance_B.txt",
                           top_k_file="0_0_top_k.txt")
            gc.collect()

        with open("0_0_mutual_distance_A.txt", "r") as mutual_file:
            avg = mutual_file.read().split("\n")
            avg = [float(e) for e in avg if e != ""]
            avg = sum(avg) / len(avg)
        logging.info("Average mutual distance A:" + str(avg))
        fw = open("average_mutual_distance_A.tsv", "a")
        fw.write("0\t0\t" + str(avg) + "\n")
        fw.close()

        with open("0_0_mutual_distance_B.txt", "r") as mutual_file:
            avg = mutual_file.read().split("\n")
            avg = [float(e) for e in avg if e != ""]
            avg = sum(avg) / len(avg)
        logging.info("Average mutual distance B:" + str(avg))
        fw = open("average_mutual_distance_B.tsv", "a")
        fw.write("0\t0\t" + str(avg) + "\n")
        fw.close()

        with open("0_0_distance.txt", "r") as distance_file:
            avg = distance_file.read().split("\n")
            avg = [float(e) for e in avg if e != ""]
            avg = sum(avg)/len(avg)
        logging.info("Average distance:" + str(avg))
        fw = open("average_distance.tsv", "a")
        fw.write("0\t0\t" + str(avg) + "\n")
        fw.close()


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(train_dataset.dataloader):  # inner loop within one epoch
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
                save_suffix = 'epoch_'+str(epoch)+'_iter_%d' % total_iters

                model.save_networks(save_suffix)
                if previous_suffix is not None:
                    model.delete_networks(previous_suffix)
                previous_suffix = save_suffix
                model.save_networks('latest')

            if opt.eval_freq is not None and total_iters % opt.eval_freq == 0:
                sentences_filename = str(epoch)+"_"+str(total_iters)+"_eval_sentences.txt"
                distance_filename = str(epoch)+"_"+str(total_iters)+"_distances.txt"
                top_k_filename = str(epoch)+"_"+str(total_iters)+"_top_k.txt"
                mutual_filename_A = str(epoch)+"_"+str(total_iters)+"_mutual_distances_A.txt"
                mutual_filename_B = str(epoch)+"_"+str(total_iters)+"_mutual_distances_B.txt"

                fw = open(distance_filename, "w")
                fw.close()
                fw = open(sentences_filename, "w")
                fw.close()
                fw = open(top_k_filename, "w")
                fw.close()
                fw = open(mutual_filename_A, "w")
                fw.close()
                fw = open(mutual_filename_A, "w")
                fw.close()
                for j, eval_data in enumerate(eval_dataset.dataloader):  # inner loop within one epoch
                    if j > 20:
                        break
                    model.set_input(eval_data)  # unpack data from dataset and apply preprocessing
                    model.evaluate(sentences_file=sentences_filename, distance_file=distance_filename, mutual_avg_file_A=mutual_filename_A, mutual_avg_file_B=mutual_filename_B,
                                   top_k_file=top_k_filename, epoch=epoch, iters=total_iters)
                with open(mutual_filename_A, "r") as mutual_file:
                    avg = mutual_file.read().split("\n")
                    avg = [float(e) for e in avg if e != ""]
                    avg = sum(avg) / len(avg)
                logging.info("Average mutual distance A:" + str(avg))
                fw = open("average_mutual_distance_A.tsv", "a")
                fw.write(str(epoch) + "\t" + str(total_iters) + "\t" + str(avg) + "\n")
                fw.close()

                with open(mutual_filename_B, "r") as mutual_file:
                    avg = mutual_file.read().split("\n")
                    avg = [float(e) for e in avg if e != ""]
                    avg = sum(avg) / len(avg)
                logging.info("Average mutual distance B:" + str(avg))
                fw = open("average_mutual_distance_B.tsv", "a")
                fw.write(str(epoch)+"\t"+str(total_iters)+"\t" + str(avg) + "\n")
                fw.close()


                with open(distance_filename, "r") as distance_file:
                    avg = distance_file.read().split("\n")
                    avg = [float(e) for e in avg if e != ""]
                    avg = sum(avg) / len(avg)
                logging.info("Average distance:" + str(avg))
                fw = open("average_distance.tsv", "a")
                fw.write(str(epoch)+"\t"+str(total_iters)+"\t" + str(avg) + "\n")
                fw.close()

            iter_data_time = time.time()
        logging.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

        sentences_filename = "eval_sentences.txt"
        distance_filename = "distances.txt"
        top_k_filename = "top_k.txt"
        mutual_filename_A = "mutual_distances_A.txt"
        mutual_filename_B = "mutual_distances_B.txt"
        with open(distance_filename, "a") as distance_file:
            distance_file.write("NEW EPOCH:\n")
        with open(top_k_filename, "a") as top_file:
            top_file.write("NEW EPOCH:\n")
        with open(sentences_filename, "a") as sentences_file:
            sentences_file.write("NEW EPOCH:\n")
        with open(mutual_filename_A, "a") as mutual_file:
            mutual_file.write("NEW EPOCH:\n")
        with open(mutual_filename_B, "a") as mutual_file:
            mutual_file.write("NEW EPOCH:\n")

        for j, eval_data in enumerate(eval_dataset.dataloader):  # inner loop within one epoch
            model.set_input(eval_data)  # unpack data from dataset and apply preprocessing
            model.evaluate(sentences_file=sentences_filename, distance_file=distance_filename, mutual_avg_file_A=mutual_filename_A, mutual_avg_file_B=mutual_filename_B, top_k_file=top_k_filename)

        with open(distance_filename, "a") as distance_file:
            distance_file.write("\n\n\n\n")
        with open(top_k_filename, "a") as top_file:
            top_file.write("\n\n\n\n")
        with open(sentences_filename, "a") as sentences_file:
            sentences_file.write("\n\n\n\n")
        with open(mutual_filename_A, "a") as mutual_file:
            sentences_file.write("\n\n\n\n")
        with open(mutual_filename_B, "a") as mutual_file:
            sentences_file.write("\n\n\n\n")

        logging.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
