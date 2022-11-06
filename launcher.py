from roboflow import Roboflow
import os
import torch
import webbrowser


#for every execution, remember to increase the "exp" folder, for examle, the 2 detection will be in (...)/runs/train/exp2/(...), the third in exp3 and so on. Same thing when showing detecting results

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
                                                            #dataset path
os.environ["DATASET_DIRECTORY"] = "/home/" +os.getlogin()+"/Desktop/LABIAGI_grapesdetector/"
rf = Roboflow(model_format="yolov5", notebook="ultralytics")

rf = Roboflow(api_key="olElazA0TsqKaw9m4wG7")
project = rf.workspace("iagi").project("iagi")
dataset = project.version(20).download("yolov5")
                                    #    img size  batch size   epochs                                              data.yaml dataset path
os.system("sudo python3 yolov5/train.py --img 640 --batch 35 --epochs 300 --data /home/"+os.getlogin()+"/Desktop/LABIAGI_grapesdetector/IAGI-20/data.yaml  --cache")
                                           #weights path                                                                              img size                                          test images path                       bbboux line thickness      min conf thresh
os.system("sudo python3 yolov5/detect.py --weights /home/"+os.getlogin()+"/Desktop/LABIAGI_grapesdetector/yolov5/runs/train/exp6/weights/best.pt  --img 640  --conf 0.1 --source /home/"+os.getlogin()+"/Desktop/LABIAGI_grapesdetector/IAGI-20/test/images --line-thickness 1 --conf-thres 0.51 ")

os.system("nautilus " +  "/home/"+os.getlogin()+"/Desktop/LABIAGI_grapesdetector/yolov5/runs/detect/exp6 &") #detected results

os.system("tensorboard --logdir /home/"+ os.getlogin()+"/Desktop/LABIAGI_grapesdetector/yolov5/runs/ &")
                #or "firefox"             
webbrowser.get("google-chrome").open("http://localhost:6006/") #shows tensorboard results

