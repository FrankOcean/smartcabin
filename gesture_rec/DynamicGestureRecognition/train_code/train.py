import argparse
import configparser
from ast import literal_eval
import os
from math import ceil
import numpy as np
import lib.image as kmg
from lib.data_loader import DataLoader
from lib.utils import mkdirs
import lib.model as model
from lib.model_res import Resnet3DBuilder
from keras.callbacks import ModelCheckpoint  # resnet使用
#from tensorflow.python.keras.callbacks import ModelCheckpoint # C3D 使用
from keras.optimizers import SGD

def main(args):
    # 从配置文件中提取信息
    #general
    nb_frames    = config.getint('general', 'nb_frames')
    skip         = config.getint('general', 'skip')
    target_size  = literal_eval(config.get('general', 'target_size'))
    batch_size   = config.getint('general', 'batch_size')
    epochs       = config.getint('general', 'epochs')
    nb_classes   = config.getint('general', 'nb_classes')
    #path
    model_name   = config.get('path', 'model_name')
    data_root    = config.get('path', 'data_root')
    data_model   = config.get('path', 'data_model')
    data_vid     = config.get('path', 'data_vid')
    path_weights = config.get('path', 'path_weights')
    csv_labels   = config.get('path', 'csv_labels')
    csv_train    = config.get('path', 'csv_train')
    csv_val      = config.get('path', 'csv_val')
    #option
    workers              = config.getint('option', 'workers')
    use_multiprocessing  = config.getboolean('option', 'use_multiprocessing')
    max_queue_size       = config.getint('option', 'max_queue_size')

    # 将需要的路径连接在一起
    path_vid = os.path.join(data_root, data_vid) # 数据集全路径，到数据集的文件名那
    path_model = os.path.join(data_root, data_model, model_name)  # 网络模型路径
    path_labels = os.path.join(data_root, csv_labels)  # 总标签路径
    path_train = os.path.join(data_root, csv_train)  # 训练集标签路径
    path_val = os.path.join(data_root, csv_val)  # 测试机标签路径

    # 输入张量的输入形状 (16, 64, 96, 3)  (batch,height,width,channles)
    inp_shape   = (nb_frames,) + target_size + (3,)
    # 使用DataLoader类加载数据
    data = DataLoader(path_vid, path_labels, path_train, path_val)
    # 创建待保存模型的文件夹
    mkdirs(path_model, 0o755)
    # 为训练和验证集创建生成器
    gen = kmg.ImageDataGenerator()
    gen_train = gen.flow_video_from_dataframe(data.train_df, path_vid, path_classes=path_labels,
                                              x_col='video_id', y_col="label",
                                              target_size=target_size, batch_size=batch_size,
                                              nb_frames=nb_frames, skip=skip, has_ext=True)
    gen_val = gen.flow_video_from_dataframe(data.val_df, path_vid, path_classes=path_labels,
                                            x_col='video_id', y_col="label",
                                            target_size=target_size, batch_size=batch_size,
                                            nb_frames=nb_frames, skip=skip, has_ext=True)

    # RESNET3D model
    # net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes, drop_rate=0.5)
    # opti = SGD(lr=0.01, momentum=0.9, decay= 0.0001, nesterov=False)
    # net.compile(optimizer=opti,
    #             loss="categorical_crossentropy",
    #             metrics=["accuracy"])

    #  CNN3D Lite 模型
    net = model.CNN3D_lite(inp_shape=inp_shape, nb_classes=nb_classes)
    net.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy", "top_k_categorical_accuracy"])

    # 如果模型权重文件存在，加载模型权重
    if(path_weights != "None"):
        print("Loading weights from : " + path_weights)
        net.load_weights(path_weights)

    # 保存最佳模型的文件格式
    model_file_format_best = os.path.join(path_model,'model.best.hdf5') 

    # 检查最佳模型
    # monitor监视器:需要监控的数量。   Verbose:详细模式，取值为0或1。
    checkpointer_best = ModelCheckpoint(model_file_format_best, monitor='val_accuracy',verbose=1, save_best_only=True,save_weights_only=True, mode='max')

    # 获得训练集和验证集中的样本数量
    nb_sample_train = data.train_df["video_id"].size
    nb_sample_val   = data.val_df["video_id"].size

    # 开始训练
    net.fit_generator(
            generator=gen_train, # 生成器函数，生成器的输出应该为: (inputs, targets)的tuple；
            steps_per_epoch=ceil(nb_sample_train/batch_size),# int，当生成器返回steps_per_epoch次数据时一个epoch结束，执行下一个epoch
            epochs=epochs,
            validation_data=gen_val,
            validation_steps=ceil(nb_sample_val/batch_size),
            shuffle=True,
            verbose=1,
            workers=workers,
            max_queue_size=max_queue_size,  # 生成器队列的最大容量
            use_multiprocessing=use_multiprocessing,
            callbacks=[checkpointer_best],
    )
    # 训练后，将最终的模型序列化为JSON
    model_json = net.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # 保存权重
    net.save_weights(model_name + ".h5")
    print("Saved model to disk")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",default="./config.cfg", help="运行脚本所需的配置文件")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    main(config)
