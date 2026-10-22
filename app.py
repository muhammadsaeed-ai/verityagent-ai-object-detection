"""CLI entry point."""
import argparse, logging
from pathlib import Path
from config import DetectionConfig
from detector import detect_objects, load_image, load_model
from visualization import annotate_image, display_image, save_image
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
LOG=logging.getLogger("verityagent-object-detection")
def build_parser():
    parser=argparse.ArgumentParser(description="Detect COCO objects in a still image using pretrained Faster R-CNN.")
    parser.add_argument("--image",type=Path,default=DetectionConfig.image_path,help="Input image path.")
    parser.add_argument("--output",type=Path,default=DetectionConfig.output_path,help="Annotated output path.")
    parser.add_argument("--threshold",type=float,default=DetectionConfig.confidence_threshold,help="Confidence threshold from 0.0 to 1.0.")
    parser.add_argument("--no-display",action="store_true",help="Save without opening a GUI.")
    return parser
def main():
    args=build_parser().parse_args()
    try:
        config=DetectionConfig(args.image,args.output,args.threshold,not args.no_display)
        image=load_image(config.image_path)
        model,weights,device=load_model()
        detections,_=detect_objects(image,model,weights,device,config.confidence_threshold)
        annotated=annotate_image(image,detections)
        saved=save_image(annotated,config.output_path)
    except (FileNotFoundError,ValueError,RuntimeError,OSError) as exc:
        LOG.error("%s",exc); return 1
    LOG.info("Device: %s",device)
    LOG.info("Detections above threshold %.2f: %d",config.confidence_threshold,len(detections))
    for d in detections: LOG.info("  %-20s confidence=%.3f box=%s",d["label"],d["score"],d["box"])
    LOG.info("Annotated image saved to: %s",saved)
    if config.display:
        try: display_image(annotated)
        except RuntimeError as exc: LOG.error("%s",exc); return 1
    return 0
if __name__=="__main__": raise SystemExit(main())
