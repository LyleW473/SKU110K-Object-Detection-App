import set_path
import torch
import yaml
from sahi import AutoDetectionModel
from sahi.predict import predict

if __name__ == "__main__":
    """
    This script is used for testing a trained YOLO model
    on the entire test dataset. It will save the output
    images in the runs/predict folder.
    """
    model_num = 8
    model_dir = f"runs/detect/train{model_num}"
    model_path = f"{model_dir}/weights/best.pt"
    print(model_path)
    detection_model = AutoDetectionModel.from_pretrained(
                                                        model_type="yolov8",
                                                        model_path=model_path,
                                                        confidence_threshold=0.3,
                                                        device="cpu" if not torch.cuda.is_available() else "cuda:0"
                                                        )
    with open(f"{model_dir}/args.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    res = predict(
                model_type="yolov8",
                model_path=model_path,
                model_device='cuda:0',
                model_confidence_threshold=0.4,
                source="data/yolo_dataset_base/test/images",
                slice_height=config["imgsz"],
                slice_width=config["imgsz"],
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                )