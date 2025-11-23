from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v10/yolov10n-PIDray.yaml')  # 指定YOLO模型对象，并加载指定配置文件中的模型配置
    # 加载预训练的权重文件'yolov10s.pt'，加速训练并提升模型性能
    model = YOLO(
        '/data/projects/ultralytics-main-neu/ultralytics-main/runs/detect/v8nGC5.10/weights/best.pt')
    model.val(data='/data/projects/ultralytics-main-neu/ultralytics-main/d_10_gc10/data.yaml',  # 指定训练数据集的配置文件路径，这个.yaml文件包含了数据集的路径和类别信息
              split='val',
              imgsz=640,
              batch=16,
              project='runs/val',
              name='v8n-GC5.10',
              )
