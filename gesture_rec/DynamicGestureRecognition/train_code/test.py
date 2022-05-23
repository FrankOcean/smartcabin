import argparse
import configparser
from ast import literal_eval

import os
from math import ceil
import numpy as np

import lib.image as kmg
from lib.data_loader import DataLoader
import lib.model as model
from lib.model_res import Resnet3DBuilder

def main(args):
    # extract information from the configuration file
    nb_frames    = config.getint('general', 'nb_frames')
    skip         = config.getint('general', 'skip')
    target_size  = literal_eval(config.get('general', 'target_size'))
    batch_size   = config.getint('general', 'batch_size')
    epochs       = config.getint('general', 'epochs')
    nb_classes   = config.getint('general', 'nb_classes')

    model_name   = config.get('path', 'model_name')
    data_root    = config.get('path', 'data_root')
    data_model   = config.get('path', 'data_model')
    data_vid     = config.get('path', 'data_vid')

    path_weights = config.get('path', 'path_weights')

    csv_labels   = config.get('path', 'csv_labels')
    csv_test     = config.get('path', 'csv_test')

    workers              = config.getint('option', 'workers')
    use_multiprocessing  = config.getboolean('option', 'use_multiprocessing')
    max_queue_size       = config.getint('option', 'max_queue_size')

    # join together the needed paths
    path_vid = os.path.join(data_root, data_vid)
    path_model = os.path.join(data_root, data_model, model_name)
    path_labels = os.path.join(data_root, csv_labels)
    path_test = os.path.join(data_root, csv_test)

    # Input shape of the input Tensor
    inp_shape   = (nb_frames,) + target_size + (3,)

    # load the data using the DataLoader class
    data = DataLoader(path_vid, path_labels, path_test=path_test)

    # create the generator for the test set
    gen = kmg.ImageDataGenerator()
    gen_test = gen.flow_video_from_dataframe(data.test_df, path_vid, shuffle=False, path_classes=path_labels, class_mode=None, x_col='video_id', target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)

    # build and compile RESNET3D model
    # net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes, drop_rate=0.5)

    # build and compile CNN3D Lite model
    net = model.CNN3D_lite(inp_shape=inp_shape, nb_classes=nb_classes)

    # if weights file is present load the weights
    if(path_weights != "None"):
        print("Loading weights from : " + path_weights)
        net.load_weights(path_weights)
    else: 
        sys.exit("<Error>: Specify a value for path_weights different from None when using test mode")

    # get the number of samples in the test set 
    nb_sample_test = data.test_df["video_id"].size

    res = net.predict_generator(
        generator=gen_test,
        steps=ceil(nb_sample_test/batch_size),
        verbose=1,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
    )

    # create an empty column called label
    data.test_df['label']=""

    # for each result get the string label and set it in the dataFrame
    for i, item in enumerate(res):
        item[item == np.max(item)]=1
        item[item != np.max(item)]=0
        label=data.categorical_to_label(item)

        data.test_df.at[i,'label'] = label

    # save the resulting dataframe to a csv
    data.test_df.to_csv(os.path.join(path_model, "prediction.csv"), sep=';', header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Configuration file to run the script", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)
