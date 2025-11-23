import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\yolov8-gai\YOLOv8-yan\ultralytics\cfg\models\v8\my-yolov8-1.yaml')
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov8s.yaml就是使用的v8s,
    # 类似某个改进的yaml文件名称为yolov8-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov8l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！

    # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度

    model.train(data=r'D:\yolov8-gai\YOLOv8-yan\ultralytics\cfg\datasets\VOC.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=1280,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=2,
                close_mosaic=0,
                workers=0,
                device='',
                optimizer='SGD', # using SGD
                # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )
