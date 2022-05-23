import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model_v4 import inception_v4
#检测单张图片
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize((448,448)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # load image
    img_path = "./321.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img=img.view(-1,3,448,448)
    # read class_indict
    json_path = './class_indices_v4.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # create model
    model = inception_v4(classes=10).to(device)
    # load model weights
    weights_path = "./googleNet_v4.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device),strict=False)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        print(predict)
        predict_cla = torch.argmax(predict).numpy()
        print(predict_cla)
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()

if __name__ == '__main__':
    main()