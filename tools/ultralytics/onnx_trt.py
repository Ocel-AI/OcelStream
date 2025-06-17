import argparse
from ultralytics import YOLO

def export_model(weights, topk_all, iou_thres, conf_thres, class_agnostic, pooler_scale, sampling_ratio, mask_resolution):
    # Initialize the model with the provided weights file
    model = YOLO(weights)
    
    # Export the model to ONNX format with TensorRT optimization
    model.export(
        format="onnx_trt", 
        dynamic=True, 
        topk_all=topk_all, 
        iou_thres=iou_thres, 
        conf_thres=conf_thres,
        class_agnostic=class_agnostic,
        pooler_scale=pooler_scale,
        sampling_ratio=sampling_ratio,
        mask_resolution=mask_resolution
    )

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX with TensorRT optimization.")
    
    # Add the -w/--weights argument to specify the weights file
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to the YOLO weights file (e.g., yolov8n.pt)')
    
    # Add other arguments with default values
    parser.add_argument('--topk_all', type=int, default=100, help='Number of top K detections for all classes (default: 100)')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='Confidence threshold for NMS (default: 0.25)')
    parser.add_argument('--class_agnostic', action='store_true', help='Set class agnostic NMS (default: False)')
    parser.add_argument('--pooler_scale', type=float, default=0.25, help='Pooler scale for ROI operations (default: 0.25)')
    parser.add_argument('--sampling_ratio', type=int, default=0, help='Sampling ratio for ROI align (default: 0)')
    parser.add_argument('--mask_resolution', type=int, default=160, help='Resolution for masks during export (default: 160)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the export_model function with the provided arguments
    export_model(
        args.weights,
        args.topk_all,
        args.iou_thres,
        args.conf_thres,
        args.class_agnostic,
        args.pooler_scale,
        args.sampling_ratio,
        args.mask_resolution
    )

if __name__ == "__main__":
    main()


