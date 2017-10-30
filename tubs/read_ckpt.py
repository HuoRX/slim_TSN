# from matplotlib import pyplot as plt
# import numpy as np
import tensorflow as tf
import argparse
import sys
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags


slim = tf.contrib.slim

#

def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors, tensor_contain, output):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
        file_name: Name of the checkpoint file.
        tensor_name: Name of the tensor in the checkpoint file to print.
        all_tensors: Boolean indicating whether to print all tensors.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                num = reader.get_tensor(key)
                print("tensor_name: ", key)
                #print(num)
                if output is not None:
                    if output=='shape':
                        print("shape:", num.shape)        # read into numpy
                    else:
                        print("value:", num)
        elif tensor_contain is not None:
            print('1')
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                num = reader.get_tensor(key)
                # print(key)
                if key.find(tensor_contain) >=0 :
                    print("tensor_name: ", key)
                    # print(key.find(tensor_contain))
                    # print(num)
                    if output is not None:
                        if output == 'shape':
                            print("shape:", num.shape)  # read into numpy
                        else:
                            print("value:", num)
        elif not tensor_name:
            print('2')
            print(reader.debug_string().decode("utf-8"))
        else:
            print('3')
            print("tensor_name: ", tensor_name)
            num = reader.get_tensor(tensor_name)
            if output is not None:
                if output == 'shape':
                    print("shape:", num.shape)  # read into numpy
                else:
                    print("value:", num)
    except Exception as e:    # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                        "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.    Try removing the '.' and extension.    Try:
inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))


def InitAssignFn(model_path):
    if tf.gfile.IsDirectory(model_path):
        checkpoint_path = tf.train.latest_checkpoint(model_path)
        print(1)
    else:
        checkpoint_path = model_path
        print(2)
    return checkpoint_path






def main():

    # model_path = '/home/jingyu/Herbert/python_project/Big-Video/ckpt/two/rgb'
    #
    # # 'build_net/nets/checkpoint/inception_v3.ckpt','data/dataset1/ckpt1/'
    # '/home/jingyu/Herbert/python_project/TSN/build_net/nets/checkpoint/inception_v3.ckpt'
    # '/home/jingyu/Herbert/python_project/TSN/data/ucf101/split1/ckpt/aa/f_a'
    # '/home/jingyu/Herbert/python_project/TSN/data/ucf101/split1/ckpt/aa/best'
    # print('ckpt path:', model_path)
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--checkpoint_dir', type=str, help="The path to save the accuracy report", default=None)
    # parser.add_argument('--tensor_name', type=str, help="The path to save the accuracy report", default=None)
    # parser.add_argument('--tensor_contain', type=str, help="The path to save the accuracy report", default=None)
    # parser.add_argument('--all_tensors', type=bool, help="The path to save the accuracy report", default=False)
    # parser.add_argument('--output', type=str, help="Print out the output, shape, value, or nothing", default=None)
    #
    # args = parser.parse_args()
    #
    # checkpoint_dir = args.checkpoint_dir
    # tensor_name = args.tensor_name
    # tensor_contain = args.tensor_contain
    # all_tensors = args.all_tensors
    # output = args.output

    checkpoint_dir = '/Users/huoruixin/Downloads/tfslim/slim_TSN/ckpt/inception_v3.ckpt'
    tensor_name = 'InceptionV2/Conv2d_1a_7x7/weights'
    tensor_contain = None
    all_tensors = True
    output = 'shape'

    checkpoint = InitAssignFn(checkpoint_dir)
    print_tensors_in_checkpoint_file(file_name=checkpoint, all_tensors=all_tensors,
                                     tensor_name=tensor_name, tensor_contain=tensor_contain,
                                     output=output)


if __name__ == '__main__':
    main()