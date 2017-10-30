from __future__ import division

from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
from mynet import nets_factory
# from preprocessing import preprocessing_factory,inception_preprocessing
import scipy.misc
import cv2
import imageio
from PIL import Image
import math
import os
import time
import argparse

slim = tf.contrib.slim



# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.


def eval_video_trimmed(model_name, ckpt_path, video_folder, label_list=None,
                       train_size=None, img_h=240,img_w=320, num_classes=None,
                       segment=25, agg_fn='average', k=None, save_path=None):

    with tf.Graph().as_default():
        image_ori = tf.placeholder(dtype=tf.float32, shape=(segment, img_h,img_w,3))

        network_fn = nets_factory.get_network_fn(
            model_name,
            num_classes=num_classes,
            is_training=False,
            segment_num=segment*10)

        input_size = train_size or network_fn.default_image_size

        # preprocessing_name = model_name
        # image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,is_training=False)
        # processed_images = image_preprocessing_fn(image_ori, image_size, image_size)

        # preprocessing
        processed_images = tf.subtract(image_ori, 0.5)
        processed_images = tf.multiply(processed_images, 2.0)
        #
        # crop 4 corners and center (with flip flop, total input shape is [10*segment, img_h, img_w, 3] for one video)
        crop_size = [segment, input_size, input_size, 3]
        crop_start = [[0, 0, 0, 0],                                                                # left top corner
                      [0, img_h - input_size, 0, 0],                                               # left bottom corner
                      [0, 0, img_w - input_size, 0],                                               # right top corner
                      [0, img_h - input_size, img_w - input_size, 0],                              # right bottom corner
                      [0, int((img_h - input_size)/2.0), int((img_w - input_size)/2.0), 0]]        # center
        cropped_batch = tf.slice(processed_images, begin=crop_start[0], size=crop_size)
        for i in range(1, 5):
            cropped_img = tf.slice(processed_images, begin=crop_start[i], size=crop_size)
            cropped_batch = tf.concat([cropped_batch, cropped_img], axis=0)

        flipped_img = tf.reshape(processed_images, [segment*img_h, img_w, 3])
        flipped_img = tf.image.flip_left_right(flipped_img)
        flipped_img = tf.reshape(flipped_img,[segment, img_h, img_w, 3])

        for i in range(5):
            cropped_img = tf.slice(flipped_img, begin=crop_start[i], size=crop_size)
            cropped_batch = tf.concat([cropped_batch, cropped_img], axis=0)

        processed_batch = tf.reshape(cropped_batch, [10*segment, input_size, input_size, 3])

        # processed_batch = tf.image.resize_image_with_crop_or_pad(processed_images, input_size, input_size)

        logits, endpoints = network_fn(processed_batch)
        logit_predict = endpoints['Predictions']
        # logits_agg = tf.reduce_mean(logits, axis=0)


        # Create a function that reads the network weights
        # from the checkpoint file that you downloaded.
        # We will run it in session later.

        if tf.gfile.IsDirectory(ckpt_path):
            # print(1)
            checkpoint_path = tf.train.latest_checkpoint(ckpt_path)
        else:
            # print(2)
            checkpoint_path = ckpt_path

        variables_to_restore = slim.get_model_variables()
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


        def predict_frame_folder(video_path, segment=25, k=None):        # video path means the folder which contains all frames extracted from that video
            frames = sorted(os.listdir(video_path))
            fcount = len(frames)  # count the number of frames
            # print('length', fcount)
            tick = np.linspace(0, fcount-1, segment)

            frame_batch = np.zeros((segment, img_h, img_w, 3))
            for i in range(segment):
                frame_num = int(tick[i])
                frame_path = os.path.join(video_path, frames[frame_num])
                # print(frame_path)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = scipy.misc.imresize(frame, [img_h,img_w,3])
                frame_batch[i] = frame/255.0

            out = sess.run(logit_predict, feed_dict={image_ori: frame_batch})          # each time input 25 frames of one video
            # print('output',out)


            # if agg_fn == 'average':
            #     # print('agg_fn is average')
            #     out = np.mean(out, axis=0)
            #     out_agg = np.argmax(out)
            #
            # elif agg_fn == 'max':
            #     # print('agg_fn is max')
            #     out = np.max(out, axis=0)
            #     out_agg = np.argmax(out)
            #
            # elif agg_fn == 'topKmax':
            #     if k is None:
            #         k = 10
            #     # print('agg_fn is topKmax, k is ',k)
            #     out = np.sort(out, axis=0)
            #     out = out[-k:]
            #     out = np.mean(out, axis=0)
            #     out_agg = np.argmax(out)
            #
            # else:
            #     # print('agg_fn is the most frequent label')
            #     out = np.argmax(out, axis=1)
            #     b, counts = np.unique(out, return_counts=True)
            #     ind = np.argmax(counts)
            #     out_agg = b[ind]


            return out
            # print('Predict:', np.argmax(out1))

        with tf.Session() as sess:
            # a = sess.run(sp, feed_dict={image_ori:image})
            # print(a.shape)

            # Load weights
            init_fn(sess)
            # sess.run(init_fn)


            if video_folder[-1] == '/':
                video_folder = video_folder[:-1]

            category_list = sorted(os.listdir(video_folder))
            category_list_2 = category_list.copy()
            for file in category_list:
                if not os.path.isdir(os.path.join(video_folder, file)):
                    # print(file)
                    category_list_2.remove(file)  # remove file, only left video folders
            category_list = category_list_2

            if num_classes is None:
                num_classes = len(category_list)

            if label_list == None:
                label_list = list(range(num_classes))

            assert num_classes == len(label_list)
            assert num_classes == len(category_list)

            out = ['category, accuracy, video_num, {}\n'.format(agg_fn)]
            print('category, accuracy, video_num, {}'.format(agg_fn))
            total_num = 0
            total_correct = 0
            for i in range(len(category_list)):
                category = category_list[i]
                label = label_list[i]
                video_path = os.path.join(video_folder,category)

                files = sorted(os.listdir(video_path))
                # print(files)
                video_num = len(files)
                total_num += video_num
                correct = 0
                for file in files:
                    video_path_in = os.path.join(video_path, file)
                    logits_ = predict_frame_folder(video_path_in, segment, k=k)
                    # print(logits_.shape)
                    label_out = np.argmax(logits_)
                    # print('label out', label_out)
                    if label_out == label:
                        correct += 1
                    # out.append('{},{},{}\n'.format(video_path_in,logits_,label_out))
                    # print('{},{}'.format(video_path_in,logits_))
                total_correct += correct
                accuracy = float(correct)/video_num
                out_line = ('{},{:.5f},{},[{}/{}]\n'.format(category, accuracy, video_num, i, (len(category_list)-1)))
                print(out_line)
                out.append(out_line)

            lastline = ('Total accuracy:{:.5f}, total number:{}'.format(float(total_correct)/total_num, total_num))
            print(lastline)
            out.append(lastline)
            if save_path is None:
                save_path = ckpt_path
            open(save_path + '/' + 'Accuracy_{}_{}.txt'.format(model_name,agg_fn), 'w').writelines(out)



            # video = imageio.get_reader(video_path)
            # fcount = video.get_length()  # count the number of readable frames
            # print('length', fcount)
            # tick = np.linspace(5, fcount-5, segment)
            #
            # l_list = np.zeros((segment,num_classes))
            # for i in range(segment):
            #     frame = video.get_data(int(tick[i]))
            #     frame = scipy.misc.imresize(frame, [img_h,img_w,3])
            #     logits_out = sess.run(logits, feed_dict={image_ori: frame/255.0})
            #
            #     l_list[i] = logits_out
            #     print(int(tick[i]))
            #
            # out1 = np.mean(l_list, axis=0)
            # print('Predict:', np.argmax(out1))



def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='inception_v3_frozen_BN', help="The model used, default is inception_v3.")
    parser.add_argument('--train_size', type=int, help="The input size for the network. 1-D integer, default is 224")
    parser.add_argument('--img_h', type=int, default=240, help="The height of input frame. 1-D integer, default is 240")
    parser.add_argument('--img_w', type=int, default=320, help="The height of input frame. 1-D integer, default is 320")
    parser.add_argument('--segment_num', type=int, default=25, help="The segments number used to predict one video. 1-D integer, default is 25")
    parser.add_argument('--num_classes', type=int, help="The number of class in the dataset.")
    parser.add_argument('--ckpt_path', type=str, help="The path of checkpoint.")
    parser.add_argument('--frame_path', type=str, help="The folder where test frames data are saved.")
    parser.add_argument('--agg_fn', type=str, choices=['average', 'max', 'topKmax'], default='average', help="The function to aggregate the prediction, default is average.")
    parser.add_argument('--k', type=int, default=10, help="The top k logits used for topKmax agg_fn, default is 10")
    parser.add_argument('--save_path', type=str, help="The path to save the accuracy report", default=None)
    
    args = parser.parse_args()
    
    model_name = args.model_name
    train_size = args.train_size
    img_h = args.img_h
    img_w = args.img_w
    segment_num = args.segment_num
    num_classes = args.num_classes
    ckpt_path = args.ckpt_path
    frame_path = args.frame_path
    agg_fn = args.agg_fn
    k = args.k
    save_path = args.save_path
    
    print(args)


    # model_name='inception_v2_frozen_BN'
    # train_size=224
    # img_h=240
    # img_w=320
    # segment_num=25
    # num_classes=2
    # ckpt_path='/home/herbert/python_project/TSN/data/dataset1/ckpt_slim/v2'
    # frame_path='/home/herbert/python_project/TSN/data/dataset1/test/frame_all'
    # agg_fn='average'
    # k=10
    # save_path=ckpt_path




    eval_video_trimmed(model_name=model_name, ckpt_path=ckpt_path, video_folder=frame_path,
                       train_size=train_size, img_h=img_h, img_w=img_w, num_classes=num_classes,
                       segment=segment_num, agg_fn=agg_fn, k=k, save_path=save_path)


    # print(model_name)
    # print(train_size)
    # print(img_h)
    # print(img_w)


if __name__ == "__main__":
    time1 = time.time()
    main()
    time2 = time.time()
    print('Time cost:',time2-time1)

