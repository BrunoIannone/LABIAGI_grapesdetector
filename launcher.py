from roboflow import Roboflow
import os
import torch



print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

#print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
os.environ["DATASET_DIRECTORY"] = "/home/" +os.getlogin()+"/Desktop/detector/"
rf = Roboflow(model_format="yolov5", notebook="ultralytics")

rf = Roboflow(api_key="olElazA0TsqKaw9m4wG7")
project = rf.workspace("iagi").project("iagi")
dataset = project.version(9).download("yolov5")
os.system("sudo python3 yolov5/train.py --img 640 --batch 3 --epochs 150 --data /home/"+os.getlogin()+"/Desktop/detector/IAGI-9/data.yaml  --cache")

os.system("sudo python3 yolov5/detect.py --weights /home/bruno/Desktop/LABIAGI_grapesdetector/yolov5/runs/train/exp/weights/best.pt  --img 640  --conf 0.1 --source /home/"+os.getlogin()+"/Desktop/detector/IAGI-9/test/images --line-thickness 1 --conf-thres 0.51 ")
os.system("nautilus " +  "/home/bruno/Desktop/LABIAGI_grapesdetector/yolov5/runs/detect/exp ")
