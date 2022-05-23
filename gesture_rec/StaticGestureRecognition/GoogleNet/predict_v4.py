import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model_v4 import inception_v4
import cv2
import time
#检测视频
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize((448, 448)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # read class_indict
    json_path = './class_indices_v4.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # create model
    model = inception_v4(classes=10).to(device)
    weights_path = "./googleNet_v4.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    #写视频
    video_writer = None
    loc_time = time.localtime()
    str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
    save_video_path = "./demo/demo_{}.mp4".format(str_time)

    #读入摄像头
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    num=0
    a=''
    b=0.1
    while True:
        ret, frame = cap.read()
        num+=1
        if frame is None:
            break
        if ret == True:
            frame=cv2.flip(frame,1)
            #if num % 2:#每两帧处理一次，第二帧显示上一帧结果
            if ret:
                frame_PIL=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = data_transform(frame_PIL)
                img=img.view(-1,3,448,448)
                model.eval()
                with torch.no_grad():
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()
                a=class_indict[str(predict_cla)]
                b=predict[predict_cla].numpy()
                #print(a)
                if b<0.5:
                    a='None'
                    frame = cv2.putText(frame, text='class: {}   prob: 0.00'.format(a), org=(10, 50),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, color=(0,255,0), thickness=2)
                else:
                    frame = cv2.putText(frame,text='class: {}   prob: {:.3}'.format(a,b),org=(10,50),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1,color=(0,0,255),thickness=2)
                cv2.imshow('shipin', frame)
            else:
                if b<0.5:
                    a='None'
                    frame = cv2.putText(frame, text='class: {}   prob: 0.00'.format(a), org=(10, 50),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1, color=(0,255,0), thickness=2)
                else:
                    frame = cv2.putText(frame,text='class: {}   prob: {:.3}'.format(a,b),org=(10,50),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1,color=(0,0,255),thickness=2)
                cv2.imshow('shipin',frame)
            #写视频
            # if video_writer is None:
            #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            #     video_writer = cv2.VideoWriter(save_video_path, fourcc, fps=20, frameSize = (frame.shape[1], frame.shape[0]))
            # video_writer.write(frame)
            if cv2.waitKey(30) & 0xFF == 27:  # esc退出
                break
    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()
    json_file.close()
if __name__ == '__main__':
    main()