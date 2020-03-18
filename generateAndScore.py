# TODO :
# all paths in parameters
# add a sleep in the http requests

import requests
import hashlib
import os

from content import kutils
from content.kutils import applications as apps
from content.kutils import model_helper as mh
from content.kutils import tensor_ops as ops
from content.kutils import generic as gen
from content.kutils import image_utils as img
from tqdm import tqdm

from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model

import pandas as pd
import numpy as np

import sys
import getopt

def save_online_person(number_of_images, output_dir):
    try:
        os.mkdir(output_dir)
    except OSError:
        print ("Creation of the directory %s failed" % output_dir)
    else:
        print ("Successfully created the directory %s " % output_dir)
    nb = 0
    p_bar = tqdm(iterable=nb, total=number_of_images)
    while nb < number_of_images:
        image = requests.get("https://thispersondoesnotexist.com/image", headers={'User-Agent': 'hello'}).content
        file_name = output_dir + hashlib.md5(image).hexdigest() + ".jpeg"
        if not os.path.isfile(file_name):
            f = open(file_name, "wb")
            f.write(image)
            nb += 1
            p_bar.update(1)
    p_bar.close()

def set_ava_scores(images_path):
    model_name = 'mlsp_wide_orig'
    input_shape = (None, None, 3)
    model_base = apps.model_inceptionresnet_pooled(input_shape)
    pre = apps.process_input[apps.InceptionResNetV2]
    # pre = lambda im: preprocess_fn(img.ImageAugmenter(img.resize_image(im, (384, 512)), remap=False).fliplr(do=False).result)
    root_path = os.getcwd() + '/content/ava-mlsp/'
    input_feats = Input(shape=(5,5,16928), dtype='float32')
    x = apps.inception_block(input_feats, size=1024)
    x = GlobalAveragePooling2D(name='final_GAP')(x)

    pred = apps.fc_layers(x, name       = 'head',
                          fc_sizes      = [2048, 1024, 256,  1],
                          dropout_rates = [0.25, 0.25, 0.5, 0],
                          batch_norm    = 2)

    model = Model(inputs  = input_feats, 
                  outputs = pred)

    gen_params = dict(batch_size    = 1,
                      data_path     = images_path,                  
                      process_fn    = pre,
                      input_shape   = input_shape,
                      outputs       = 'MOS', 
                      fixed_batches = False)

    helper = mh.ModelHelper(model, model_name, pd.DataFrame(), 
                            gen_params = gen_params)

    # load head model
    helper.load_model(model_name = root_path + \
                                   'models/irnv2_mlsp_wide_orig/model')

    # join base and head models
    helper.model = Model(inputs  = model_base.input, 
                         outputs = model(model_base.output))

    for one_image in os.listdir(images_path):
        # load, pre-process it, and pass it to the model
        image_full_path = os.path.join(images_path, one_image)
        one_img = pre(img.read_image(image_full_path))
        one_img = np.expand_dims(one_img, 0)
        one_img_score = helper.model.predict(one_img)
        new_image_name = os.path.splitext(image_full_path)[0] + '_scoreAVA_' + str(one_img_score[0][0]) + os.path.splitext(image_full_path)[1]
        os.rename(image_full_path, new_image_name)

def set_koncept512_scores(images_path):
    model_name = os.getcwd() + '/content/models/KonCept512/k512'
    pre = lambda im: preprocess_fn(img.ImageAugmenter(img.resize_image(im, (384, 512)), remap=False).fliplr(do=False).result)
    # build scoring model
    base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
    head = apps.fc_layers(base_model.output, name='fc', 
                          fc_sizes      = [2048, 1024, 256, 1], 
                          dropout_rates = [0.25, 0.25, 0.5, 0], 
                          batch_norm    = 2)    

    model = Model(inputs = base_model.input, outputs = head)

    gen_params = dict(batch_size  = 32, 
                      data_path   = images_path,
                      process_fn  = pre, 
                      input_shape = (384,512,3),
                      outputs     = ('MOS',))

    # Wrapper for the model, helps with training and testing
    helper = mh.ModelHelper(model, 'KonCept512', pd.DataFrame(), 
                         loss='MSE', metrics=["MAE", ops.plcc_tf],
                         monitor_metric = 'val_loss', 
                         monitor_mode   = 'min', 
                         multiproc   = True, workers = 5,
                         gen_params  = gen_params)


    helper.load_model(model_name=model_name)

    for one_image in os.listdir(images_path):
        # load, pre-process it, and pass it to the model
        image_full_path = os.path.join(images_path, one_image)
        one_img = pre(img.read_image(image_full_path))
        one_img = np.expand_dims(one_img, 0)
        one_img_score = helper.model.predict(one_img)
        new_image_name = os.path.splitext(image_full_path)[0] + '_scoreK512_' + str(one_img_score[0][0]) + os.path.splitext(image_full_path)[1]
        os.rename(image_full_path, new_image_name)

def main(argv):
    nb_of_images = -1
    koncept_directory = ''
    ava_directory = ''
    output_dir=''

    try:
        opts, args = getopt.getopt(argv,'o:c:k:a:h',['output_images=','generate=','ava_inputs=','koncept_inputs=', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-c', '--generate'):
            nb_of_images = arg
        elif opt in ("-k", "--koncept_inputs"):
            koncept_directory = arg
        elif opt in ("-a", "--ava_inputs"):
            ava_directory = arg
        elif opt in ("-o", "--output_images"):
            output_dir = arg
        elif opt in ("-h", "--help"):
            usage()

    if (len(opts) == 0):
        usage()
        sys.exit(2)
    if (nb_of_images != -1 and output_dir == ''):
        usage()
        sys.exit(2)

    if (nb_of_images != -1):
        if (output_dir[-1] != '/'):
            output_dir += '/'
        if (output_dir[0] == '/'):
            save_online_person(number_of_images=int(nb_of_images), output_dir=output_dir)
        else:
            output_dir = os.getcwd() + '/' + output_dir
            save_online_person(number_of_images=int(nb_of_images), output_dir=output_dir)

    if (koncept_directory != ''):
        set_koncept512_scores(images_path=koncept_directory)
    if (ava_directory != ''):
        set_ava_scores(images_path=ava_directory)



def usage():
    # To generate and get number_of_images from thispersondoesnotexists:
    print("python generateAndScore.py --generate=number_of_images --output_images=output_folder")
    # To evaluate thx to koncept512 the images from folder data_input:
    print("python generateAndScore.py --koncept_inputs=data_input")
    # To evaluate thx to ava the images from folder data_input:
    print("python generateAndScore.py --ava_inputs=data_input")
    # To do everything:
    print("python generateAndScore.py --generate=number_of_images --output_images=myimages --koncept_inputs=myimages --ava_inputs=myimages")


if __name__ == '__main__':
    main(sys.argv[1:])
