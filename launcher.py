from roboflow import Roboflow
import os
import torch

os.system("pip install -r yolov5/requirements.txt")
print("GIOVANNI"+ str(torch.cuda.device_count()))
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

#print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
os.environ["DATASET_DIRECTORY"] = "/home/" +os.getlogin()+"/Desktop/detector/"
rf = Roboflow(model_format="yolov5", notebook="ultralytics")
rf = Roboflow(api_key="olElazA0TsqKaw9m4wG7")
project = rf.workspace("iagi").project("iagi")
dataset = project.version(8).download("yolov5")
print({dataset.location})
print(os.getlogin())
os.system("sudo python3 yolov5/train.py --img 640 --batch 3 --epochs 5 --data /home/"+os.getlogin()+"/Desktop/detector/IAGI-8/data.yaml  --cache")

