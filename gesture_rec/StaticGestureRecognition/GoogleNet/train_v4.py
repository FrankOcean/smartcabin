import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data
from model_v4 import inception_v4
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    batch_size = 2 #批处理大小
    lr = 0.0003     #学习率
    epochs = 50     #迭代次数
    best_acc = 0.0  #最好的精度
    save_path = './googleNet_attion_v4.pth' #保存的模型

    data_transform = {
        "train": transforms.Compose([transforms.Resize((448, 448)),transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((448, 448)),transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    image_path = os.path.join(data_root, "DataSet", "hand_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    #数据集加载
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                                               shuffle=True,num_workers=0)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size,
                                                  shuffle=False,num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    # 获取分类与下标
    hand_list = train_dataset.class_to_idx#获取train路径下的文件名+下标 为字典形式
    cla_dict = dict((val, key) for key, val in hand_list.items())
    # 写入json文件
    json_str = json.dumps(cla_dict, indent=9)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    net = inception_v4(classes=10)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            #print(labels)
            optimizer.zero_grad()
            logits= net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss = loss0
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)
        #验证
        net.eval()
        acc = 0.0  # = number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('Finished Training')

if __name__ == '__main__':
    main()