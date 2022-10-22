import sys
sys.path.insert(1, 'yolov7')
import torch 
from models.yolo import Model
from utils.torch_utils import select_device

def load_model(model_path='path/to/model.pt', conf=0.25, iou=0.5, autoshape=True):
	model = torch.load(model_path, map_location=torch.device('cpu')) if isinstance(model_path, str) else model_path 

	if isinstance(model, dict):
		model = model['ema' if model.get('ema') else 'model']  # load model

	hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
	hub_model.load_state_dict(model.float().state_dict())  # load state_dict
	hub_model.names = model.names  # class names
	if autoshape:
		hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS

	hub_model.conf = conf
	model.iou = iou
	device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
	return hub_model.to(device)