"""
	Run inference on LRM image of Viana do Castelo district with YOLOv5, Mask R-CNN and CNN using a sliding window.

	Uses:
		- TensorFlow, Keras, Pytorch, Cuda, OpenCV, PIL image, numpy...
		- ultralytics/yolov5 code https://github.com/ultralytics/yolov5.git
		- matterport/Mask_RCNN code https://github.com/matterport/Mask_RCNN.git

	@Author: Fabricio Botelho 
    @Date: 28/06/2022
    @Project: Odyssey, "Identification of archaeological sites in aerial image using deep learning"

"""

import os 
import cv2
import PIL
from PIL import Image
import glob
from pathlib import Path
import time
import pandas as pd
import numpy as np

import tensorflow.compat.v1 as tf
# Import relevant libraries
from tensorflow import keras 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
#from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import plot_model
from keras.utils import np_utils
from keras.models import load_model

import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from models.experimental import attempt_download, attempt_load
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

def convert_GeoCoord_to_PixelCoord(geocoordinate, m1, m2, b1, b2):
	"""
		Convertion of geocoordinates to pixel coordinates 

		:param geocoordinate: is a list with latitude and longitude
		:param m1: is a declive witch allow the convertion of horizontal axis
		:param m2: is a declive witch allow the convertion of vertical axis
		:param b1: is the intercept witch allow the convertion of horizontal axis 
		:param b2: is the intercept witch allow the convertion of vertical axis 
		:return: pixel coordinate x and y
	"""
	x_geo = geocoordinate[0]
	y_geo = geocoordinate[1]
	x = round(x_geo * m1 + b1)
	y = round(y_geo * m2 + b2)

	return x, y

def convert_PixelCoord_to_GeoCoord(pixelcoordinate, m1, m2, b1, b2):
	"""
		Convertion of pixel to geocoordinates 

		:param pixelcoordinate: is a list with pixel coordinates
		:param m1: is a declive witch allow the convertion of horizontal axis
		:param m2: is a declive witch allow the convertion of vertical axis
		:param b1: is the intercept witch allow the convertion of horizontal axis 
		:param b2: is the intercept witch allow the convertion of vertical axis 
		:return: geocoodinate x_geo and y_geo
	"""
	x = pixelcoordinate[0]
	y = pixelcoordinate[1]
	x_geo = (x - b1)/m1
	y_geo = (y - b2)/m2

	return x_geo, y_geo

def CountNumberOfMamoas(Xmin, Ymin, Xmax, Ymax, IoU, xywh):
	"""
		Count the number of archaeological objects (mamoas) inside a block or image or read  

		:param Xmin: minimum of coordinate on horizontal axis
		:param Ymin: minimum of coordinate on vertical axis
		:param Xmax: maximum of coordinate on horizontal axis
		:param Ymax: maximum of coordinate on vertical axis
		:return: number of objects inside of image and the resepctives center coordinates of objects
	"""

	count = 0    
	aaa = 0
	bbb = 0
  	#read mamoas of arcos (51 mamoas)
	for i in arcos_px:
		x = i[0]
		y = i[1]
		box2 = [x,y,20,20]

		if IoU:
			iou = bbox_iou(box2, xywh)
			if iou >= 0.25:
				count += 1
		else:
			if x >= Xmin and x <= Xmax and y >= Ymin and y <= Ymax:
				count += 1
				aaa = x
				bbb = y
			

  	#read mamoas of laboreiro (85 mamoas)
	for i in lab_px:
		x = i[0]
		y = i[1]
		box2 = [x,y,20,20]
		
		if IoU:
			iou = bbox_iou(box2, xywh)
			if iou >= 0.25:
				#print("IoU:---------------------- ", iou)
				count += 1
		else:
			if x >= Xmin and x <= Xmax and y >= Ymin and y <= Ymax:
				count += 1
				aaa = x
				bbb = y

	return count, aaa, bbb

def resultYOLO(results, model, path, im, im0s, img, s):
	"""
		This function serves to get the number of objects detected and the respective pixel coodinates on image
		Is based on YOLOv5 code.

		:param: results: first results before use non_max_suppression funtion
		:param: models: the model used on inference
		:param: path: directory of project
		:param: im: pixel values
		:param: im0s: pixel values
		:param: img: pixel values
		:param: s: is a string 
		:return: number of objects and respective coordinates
	"""
	results = non_max_suppression(results[0], 0.25, 0.45, None, False, 1000)

	n_objects = 0
	seen = 0
	names = model.names
	for i, det in enumerate(results):  # per image
		seen += 1
		p, im0, frame = path, im0s.copy(), getattr(img, 'frame', 0)
		X=[]
		Y=[]
		p = Path(p)  # to Path
		#txt_path = str(save_dir + p.stem) + ('' if img.mode == 'image' else f'_{frame}')  # im.txt
		s += '%gx%g ' % im.shape[2:]  # print string
		gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
		annotator = Annotator(im0, line_width=3, example=str(names))
		if len(det):
			# Rescale boxes from img_size to im0 size
			det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

			# Print results
			for c in det[:, -1].unique():
				n = (det[:, -1] == c).sum()  # detections per class
				s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

			n_objects = len(det)
            # Write results
			for *xyxy, conf, cls in reversed(det):
				xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
				X.append(xywh[0] * im.shape[2])
				Y.append(xywh[1] * im.shape[2])
				#print(X, Y)

				#line = (cls, *xywh, conf)
				#with open(f'{txt_path}.txt', 'a') as f:
					#f.write(('%g ' * len(line)).rstrip() % line + '\n')
				#c = int(cls)  # integer class
				#label = f'{names[c]} {conf:.2f}'
				#annotator.box_label(xyxy, label, color=colors(c, True))

		#if img.mode == 'image':
			#cv2.imwrite(save_dir + "/shapes/ola.jpg", im0)
	return n_objects, X, Y

class MamoasConfig(Config):
    """
    	Configuration of model to use on inference.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BATCH_SIZE = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # define initial learning rate
    LEARNING_RATE = 0.001

class InferenceMRCNNConfig(MamoasConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'object'

class Detect:
	"""
		This classe uses three function to do the detect with different algoritms, YOLOv5, Mask R-CNN and CNN
	"""

	def detectYOLO(self, ROOT_DIR, step):
		"""
			This function performs the inference on LRM image with trained CNN model
			:param: ROOT_DIR: directory of the project
			:param: step: percentage (0-100%) of sliding of window
		"""

		start_time = time.time()
		print("Running YOLOv5 detector...")

		weights = []
		for weight in glob.glob(ROOT_DIR + "*.pt"):
			weights.append(weight)
		print(weights)	

		device = torch.device('cuda', 0)
		model = attempt_load(weights, device = device)
		model.conf = 0.53 # confidence value

		dim = 640 # 40 pixels => 20x20 meters
		h = dim
		w = dim 
		slide = dim * step/100
		dims = (dim,dim)

		countObj = 0
		out = []
		tp, fp, tn, fn = 0, 0, 0, 0
		possible_geocoord = []
		col = 0

		# create a box
		x11 = 0
		x12 = 0
		x21 = dim
		x22 = 0
		x31 = dim
		x32 = dim
		x41 = 0
		x42 = dim

		xmin=0
		xmax=dim
		ymin=0
		ymax=dim

		i = 0
		for f in glob.glob(ROOT_DIR + "*.tif"):
			print(f)
			print("Loading image", Path(f).stem)
			PIL.Image.MAX_IMAGE_PIXELS = None
			image = 0
			image = Image.open(f)
			print("Uploaded image")

			width_im, height_im = image.size
			print("width:",width_im,",height:", height_im)

			rows = round(((height_im)/dim) / (step/100))
			columns = round(((width)/dim) / (step/100))
			print("rows, columns: ", rows, columns)
			col = col + columns
			i, j = 0, 0
			if Path(f).stem == "LRM_image2": # delete the imaginary offset of overlap on images tif
				x12 = x12 - 40
				x22 = x22 - 40
				x42 = x42 - 40
				x32 = x32 - 40
				i, j = 0, 0
				xmin=0
				xmax=dim
				ymin=0
				ymax=dim

			for i in range(rows):
				for j in range(columns):
					#print(i, "_", j)
					img_cropped = image.crop((xmin, ymin, xmax, ymax))
					img_cropped.save(ROOT_DIR+"_.jpg")
					numpyarray = np.array(img_cropped)

					if (np.mean(numpyarray)) != 0:
						img = LoadImages(ROOT_DIR+"_.jpg")

						for path, im, im0s, vid_cap, s in img:
							im = im
						im = torch.from_numpy(im).to(device)
						im.float()
						im = im / 255  # 0 - 255 to 0.0 - 1.0
						if len(im.shape) == 3:
							im = im[None]  # expand for batch dim

						n, X, Y, ww, hh = resultYOLO(model(im), model, path, im, im0s, img, s)
						#print(n, X, Y,ww,hh)
						nMamoas = 0
						nDeteted = 0
						countObj = countObj + n
						T_p = 0

						nMamoas_total, aaa, bbb = CountNumberOfMamoas(x11, x12, x31, x32, False, None) # total number of mamoas in this block (640x640)

						for k in range(n): # analyse objects detected
							cx = X[k] + x11
							cy = Y[k] + x12
							b = [cx,cy,ww[k],hh[k]]
							nDeteted,_,_ = CountNumberOfMamoas(cx-20, cy-20, cx+20, cy+20, False, b)
	
							if nDeteted >= 1:
								#iii = cv2.rectangle(numpyarray, (int(X[k])-20, int(Y[k])-20), (int(X[k])+20, int(Y[k])+20), (255, 0, 0), 2)
								#cv2.imwrite(ROOT_DIR + "/samples/"+ str(i)+"_"+str(x11) + "_" + str(x12) + "__" + str(x31) + str(x32) + "_" +str(n) + ".jpg", iii)
								tp+=1
								T_p+=1
								#print("tp:", tp)
								cx = X[k] + x11
								cy = Y[k] + x12
								coord = convert_PixelCoord_to_GeoCoord([cx, cy], m1, m2, b1, b2)
								out.append([str(j)+"_"+str(i)+".jpg", 0, True, cx, cy, coord])
								possible_geocoord.append([str(j)+"_"+str(i)+".jpg", coord])
							else:
								#iii = cv2.rectangle(numpyarray, (int(X[k])-20, int(Y[k])-20), (int(X[k])+20, int(Y[k])+20), (255, 0, 0), 2)
								#cv2.imwrite(ROOT_DIR + "/samples/"+ str(i)+"_"+str(x11) + "_" + str(x12) + "__" + str(x31) +"_"+ str(x32) + "_" +str(n) + ".jpg", iii)
								fp+=1
								#print("fp:", fp)
								cx = X[k] + x11
								cy = Y[k] + x12
								coord = convert_PixelCoord_to_GeoCoord([cx, cy], m1, m2, b1, b2)
								out.append([str(j)+"_"+str(i)+".jpg", 0, False, cx, cy, coord])
								possible_geocoord.append([str(j)+"_"+str(i)+".jpg", coord])

						fn = fn + (nMamoas_total - T_p)

					x11 = x11 + slide
					xmin = xmin + slide
					x12 = x12
					x21 = x21 + slide
					x22 = x22
					x41 = x41 + slide
					x42 = x42 
					x31 = x31 + slide
					xmax = xmax + slide
					x32 = x32

				x11 = 0
				xmin = 0
				x21 = dim
				x41 = 0
				x31 = dim
				xmax = dim

				x12 = x12 + slide
				ymin = ymin + slide
				x22 = x22 + slide
				x42 = x42 + slide
				x32 = x32 + slide
				ymax = ymax + slide
				j = 0

				#os.system('cls')
				print("--- %s seconds" % (time.time() - start_time))
				print("Detected: ", countObj, " mamoas.")
				print("True positives: ",tp)
				print("False positives: ",fp)
				print("False negative: ",fn)
				print("Row: ", i)
		print("Detected: ", countObj, " mamoas.")
		print("True positives: ",tp)
		print("False positives: ",fp)
		print("False negative: ",fn)

		name = []
		prediction = []
		conclusion = []
		coord_px = []
		coord_py = []
		coord_geo = []

		for i,j,k,xx,yy,geo in out:
		    name.append(i)
		    prediction.append(j)
		    conclusion.append(k)
		    coord_px.append(xx)
		    coord_py.append(yy)
		    coord_geo.append(geo)

		df = pd.DataFrame({'imageID': name,
		               	'predict(0=mamoa, 1=not_mamoa)': prediction,
		               	'conclusion': conclusion,
		               	'x': coord_px,
		               	'y': coord_py,
		               	'geo': coord_geo})
		df.to_csv(ROOT_DIR+"result_yolo.csv")
		print(ROOT_DIR+"result_yolo.csv --saved")

		cpx=coord_px
		cpy=coord_py
		c = 0
		r1, r2 = 0, 0
		a = len(name)
		while r1 < len(cpx):	
			#print(r1)
			xmin = cpx[r1]-20
			xmax = cpx[r1]+20
			ymin = cpy[r1]-20
			ymax = cpy[r1]+20

			a = len(name)
			r2 = 0
			while r2 < a:
				if r1 != r2:
					nn = name[r2]
					pr = prediction[r2]
					con = conclusion[r2]
					x = coord_px[r2]
					y = coord_py[r2]
					cg = coord_geo[r2]
					
					if x>=xmin and x<=xmax and y>=ymin and y<=ymax:
						# delete this index, because this object is detected...
						name.remove(nn)
						prediction.remove(pr)
						conclusion.remove(con)
						coord_px.remove(x)
						coord_py.remove(y)
						coord_geo.remove(cg)
				a = len(name)
				r2+=1
			r1+=1

		df = pd.DataFrame({'imageID': name,
							'predict(0=mamoa, 1=not_mamoa)': prediction,
							'conclusion': conclusion,
							'x': coord_px,
							'y': coord_py,
							'geo': coord_geo})
		df.to_csv(ROOT_DIR+"result_cnn_filtered.csv")
		print(ROOT_DIR+"result_cnn_filtered.csv --saved")



	def detectCNN(self, ROOT_DIR, step):
		"""
			This function performs the inference on LRM image with trained YOLOv5 model
			:param: ROOT_DIR: directory of the project
			:param: step: percentage (0-100%) of sliding of window
		"""

		start_time = time.time()
		print("Running CNN predict...")

		model = load_model(ROOT_DIR + "model.h5")
		print("Uploaded model")

		dim = 40 # 40 pixels => 20x20 meters
		h = dim
		w = dim 
		slide = dim * step/100
		dims = (dim,dim)

		countObj = 0
		out = []
		tp, fp, tn, fn = 0, 0, 0, 0
		possible_geocoord = []
		col = 0

		# create a box
		x11 = 0
		x12 = 0
		x21 = dim
		x22 = 0
		x31 = dim
		x32 = dim
		x41 = 0
		x42 = dim

		xmin=0
		xmax=dim
		ymin=0
		ymax=dim

		for f in glob.glob(ROOT_DIR + "*.tif"):
			print("Loading image", Path(f).stem)
			PIL.Image.MAX_IMAGE_PIXELS = None
			image = Image.open(f)
			print("Uploaded image")

			width, height = image.size
			print("width:",width,",height:", height)

			rows = round(((height)/dim) / (step/100))
			columns = round(((width)/dim) / (step/100))
			print("rows, columns: ", rows, columns)
			col = col + columns
			i, j = 0, 0
			if Path(f).stem == "LRM_image2": # delete the offset of overlap on tif images
				x12 = x12 - 40
				x22 = x22 - 40
				x42 = x42 - 40
				x32 = x32 - 40 
				i, j = 0, 0
				xmin=0
				xmax=dim
				ymin=0
				ymax=dim
			for i in range(rows):
				print("row: ", i)
				print("tp:",tp)
				print("fp:",fp)
				print("fn:",fn)
				for j in range(columns):
					img_cropped = image.crop((xmin, ymin, xmax, ymax))

					#img_cropped.save(ROOT_DIR+"_.jpg")
					numpyarray = np.array(img_cropped)

					if (np.mean(numpyarray)) != 0:
						nMamoas = 0
						nMamoas,_,_ = CountNumberOfMamoas(x11, x12, x31, x32, False, None)    
		        		
						array = np.array(img_cropped).reshape(-1, h, w, 1)
						array = array/255 # normalization
		        
						a_mean= np.mean(array)
						if (a_mean != 0):
							pred = model.predict(array)
							#pred=np.argmax(pred, axis=1)
							if pred[0] < 0.5: # (0 = mamoa <-> 1 = Not mamoa) 
								countObj+=1
								#print(countObj)
								#plt.imshow(array, cmap='gray')
								#img_cropped.save(ROOT_DIR + "images/" +str(i)+"_"+str(col)+".tif") 
								if nMamoas > 0:
									tp+=1
									print("tp:",tp)
									cx = x31-dim/2 # to get the center coordinate in block
									cy = x32-dim/2 # to get the center coordinate in block
									coord = convert_PixelCoord_to_GeoCoord([cx, cy], m1, m2, b1, b2)
									out.append([str(j)+"_"+str(i)+".jpg", 0, True, cx, cy, coord])
									possible_geocoord.append([str(j)+"_"+str(i)+".jpg",coord])
								else:
									fp+=1
									cx = x31-dim/2 # to get the center coordinate in block
									cy = x32-dim/2 # to get the center coordinate in block
									coord = convert_PixelCoord_to_GeoCoord([cx, cy], m1, m2, b1, b2)
									out.append([str(j)+"_"+str(i)+".jpg", 0, False, cx, cy, coord])
									possible_geocoord.append([str(j)+"_"+str(i)+".jpg",coord])
							else:
								#im = Image.fromarray(resized)
								#im.save("C:\\Users\\fabri\\Downloads\\folder data\\scan\\not_mamoa\\"+str(j)+"_"+str(i)+".jpg")   
								if nMamoas == 0:
									cx = x31-dim/2 # to get the center coordinate in block
									cy = x32-dim/2 # to get the center coordinate in block
									coord = convert_PixelCoord_to_GeoCoord([cx, cy], m1, m2, b1, b2)
									out.append([str(j)+"_"+str(i)+".jpg", 1, True, cx, cy, coord])
									tn+=1
								if nMamoas > 0:
									cx = x31-dim/2 # to get the center coordinate in block
									cy = x32-dim/2 # to get the center coordinate in block
									coord = convert_PixelCoord_to_GeoCoord([cx, cy], m1, m2, b1, b2)
									out.append([str(j)+"_"+str(i)+".jpg", 1, False, cx, cy, coord])
									fn+=1
									print("fn:",fn)
					x11 = x11 + slide
					xmin = xmin + slide
					x12 = x12
					x21 = x21 + slide
					x22 = x22
					x41 = x41 + slide
					x42 = x42 
					x31 = x31 + slide
					xmax = xmax + slide
					x32 = x32

				x11 = 0
				xmin = 0
				x21 = dim
				x41 = 0
				x31 = dim
				xmax = dim

				x12 = x12 + slide
				ymin = ymin + slide
				x22 = x22 + slide
				x42 = x42 + slide
				x32 = x32 + slide
				ymax = ymax + slide

				j=0

				#print("--- %s seconds" % (time.time() - start_time))
				os.system('cls')
		print("--- %s seconds" % (time.time() - start_time))
		print("Detected: ", countObj, " mamoas.")
		print("True positives: ",tp)
		print("False positives: ",fp)
		print("True negatives: ",tn)
		print("False negative: ",fn)

		name = []
		prediction = []
		conclusion = []
		coord_px = []
		coord_py = []
		coord_geo = []

		for i,j,k,xx,yy,geo in out:
		    name.append(i)
		    prediction.append(j)
		    conclusion.append(k)
		    coord_px.append(xx)
		    coord_py.append(yy)
		    coord_geo.append(geo)

		df = pd.DataFrame({'imageID': name,
		               	'predict(0=mamoa, 1=not_mamoa)': prediction,
		               	'conclusion': conclusion,
		               	'x': coord_px,
		               	'y': coord_py,
		               	'geo': coord_geo})
		df.to_csv(ROOT_DIR+"result_cnn.csv")
		print(ROOT_DIR+"result_cnn.csv --saved")

		TP = 0
		pred = prediction
		for i in range(len(name)):
			count_true = 0
			if prediction[i] == 0: # 0 means mamoa
				count_true += 1
				if prediction[i+1] == 0:
					count_true += 1
				if prediction[i-1] == 0:
					count_true += 1
				if prediction[i+col] == 0:
					count_true += 1
				if prediction[i-col] == 0:
					count_true += 1
				if prediction[i+col+1] == 0:
					count_true += 1
				if prediction[i+col-1] == 0:
					count_true += 1
				if prediction[i-col+1] == 0:
					count_true += 1
				if prediction[i-col-1] == 0:
					count_true += 1
			if count_true >= 5:
				pred[i] = 0
				TP += 1
			else:
				pred[i] = 1 
		df = pd.DataFrame({'imageID': name,
		               	'predict(0=mamoa, 1=not_mamoa)': pred,
		               	'conclusion': conclusion,
		               	'x': coord_px,
		               	'y': coord_py,
		               	'geo': coord_geo})
		df.to_csv(ROOT_DIR+"result_cnn_filtered.csv")
		print(ROOT_DIR+"result_cnn_filtered.csv --")
		print("Number of mamoas detected (filtered): ",TP)



	def detectMRCNN(self, ROOT_DIR, step):
		"""
			This function performs the inference on LRM image with trained Mask R-CNN model
			:param: ROOT_DIR: directory of the project
			:param: step: percentage (0-100%) of sliding of window
		"""

		start_time = time.time()
		print("Running Mask R-CNN detector...")

		inference_config = InferenceMRCNNConfig()

		model_path = ROOT_DIR + "model_mrcnn.h5"

		# Recreate the model in inference mode
		model = modellib.MaskRCNN(mode="inference", 
                          			config=inference_config,
                          			model_dir=model_path)

		# Get path to saved weights
		# Either set a specific path or find last trained weights
		# model_path = os.path.join(ROOT_DIR, ".h5 file name here")

		# Load trained weights
		print("Loading weights from ", model_path)
		model.load_weights(model_path, by_name=True)

		dim = 640 # 40 pixels => 20x20 meters
		h = dim
		w = dim 
		slide = dim * step/100
		dims = (dim,dim)

		countObj = 0
		out = []
		tp, fp, tn, fn = 0, 0, 0, 0
		possible_geocoord = []
		col = 0

		# create a box
		x11 = 0
		x12 = 0
		x21 = dim
		x22 = 0
		x31 = dim
		x32 = dim
		x41 = 0
		x42 = dim

		xmin=0
		xmax=dim
		ymin=0
		ymax=dim
		i=0

		for f in glob.glob(ROOT_DIR + "*.tif"):
			print("Loading image", Path(f).stem)
			PIL.Image.MAX_IMAGE_PIXELS = None
			image = Image.open(f)
			print("Uploaded image")

			width, height = image.size
			print("width:",width,",height:", height)

			rows = round(((height)/dim) / (step/100))
			columns = round(((width)/dim) / (step/100))
			print("rows, columns: ", rows, columns)
			col = col + columns

			i, j = 0, 0
			if Path(f).stem == "LRM_image2": # tirar o offset da sobreposição das imagens tif
				x12 = x12 - 40
				x22 = x22 - 40
				x42 = x42 - 40
				x32 = x32 - 40 
				i, j = 0, 0
				xmin=0
				xmax=dim
				ymin=0
				ymax=dim

			for i in range(rows):
				for j in range(columns):
					img_cropped = image.crop((xmin, ymin, xmax, ymax))
					numpyarray = np.array(img_cropped)
					if (np.mean(numpyarray)) != 0:
						img_cropped.save(ROOT_DIR+"_.jpg")
						img_cropped = cv2.imread(ROOT_DIR+"_.jpg")

						nMamoas_total, aaa, bbb = CountNumberOfMamoas(x11, x12, x31, x32, False, None)    
		        		
						results = model.detect([img_cropped])   
						r = results[0]
						X,Y = [], []
						for val in range(len(r['rois'])):
							y1, x1, y2, x2 = r['rois'][val]
							xx = ((x2-x1)/2) + x1
							yy = ((y2-y1)/2) + y1
							X.append(xx)
							Y.append(yy)
							nMamoas = 0
						nDeteted = 0
						countObj = countObj + len(r['rois'])
						T_p = 0

						for k in range(len(X)): # analyse objects detected
							cx = X[k] + x11
							cy = Y[k] + x12
							nDeteted,_,_ = CountNumberOfMamoas(cx-20, cy-20, cx+20, cy+20, False, None)
							#if nMamoas_total >= 1:
								#iii = cv2.rectangle(img_cropped, (int(X[k])-20, int(Y[k])-20), (int(X[k])+20, int(Y[k])+20), (0, 0, 255), 2)
								#iii = cv2.rectangle(iii, (aaa-x11-20, bbb-x12-20), (aaa+20-x11, bbb+20-x12), (0, 255, 0), 2)
								#cv2.imwrite(ROOT_DIR + "/samples/"+ str(cx)+"_"+str(cy)+"-"+str(aaa)+"_"+str(bbb) + ".jpg", iii)

							if nDeteted >= 1:
								tp+=1
								T_p+=1
								cx = X[k] + x11
								cy = Y[k] + x12
								coord = convert_PixelCoord_to_GeoCoord([cx, cy], m1, m2, b1, b2)
								out.append([str(j)+"_"+str(i)+".jpg", 0, True, cx, cy, coord])
								possible_geocoord.append([str(j)+"_"+str(i)+".jpg", coord])
							else:
								fp+=1
								cx = X[k] + x11
								cy = Y[k] + x12
								coord = convert_PixelCoord_to_GeoCoord([cx, cy], m1, m2, b1, b2)
								out.append([str(j)+"_"+str(i)+".jpg", 0, False, cx, cy, coord])
								possible_geocoord.append([str(j)+"_"+str(i)+".jpg", coord])

						fn = fn + (nMamoas_total - T_p)
					
					x11 = x11 + slide
					xmin = xmin + slide
					x12 = x12
					x21 = x21 + slide
					x22 = x22
					x41 = x41 + slide
					x42 = x42 
					x31 = x31 + slide
					xmax = xmax + slide
					x32 = x32

				x11 = 0
				xmin = 0
				x21 = dim
				x41 = 0
				x31 = dim
				xmax = dim

				x12 = x12 + slide
				ymin = ymin + slide
				x22 = x22 + slide
				x42 = x42 + slide
				x32 = x32 + slide
				ymax = ymax + slide
				j = 0

				print("--- %s seconds" % (time.time() - start_time))
				print("Detected: ", countObj, " mamoas.")
				print("True positives: ",tp)
				print("False positives: ",fp)
				print("False negative: ",fn)
				print("Row: ", i)
		print("Detected: ", countObj, " mamoas.")
		print("True positives: ",tp)
		print("False positives: ",fp)
		print("False negative: ",fn)

		name = []
		prediction = []
		conclusion = []
		coord_px = []
		coord_py = []
		coord_geo = []

		for i,j,k,xx,yy,geo in out:
		    name.append(i)
		    prediction.append(j)
		    conclusion.append(k)
		    coord_px.append(xx)
		    coord_py.append(yy)
		    coord_geo.append(geo)

		df = pd.DataFrame({'imageID': name,
		               	'predict(0=mamoa, 1=not_mamoa)': prediction,
		               	'conclusion': conclusion,
		               	'x': coord_px,
		               	'y': coord_py,
		               	'geo': coord_geo})
		df.to_csv(ROOT_DIR+"result_mrcnn.csv")
		print(ROOT_DIR+"result_mrcnn.csv --saved")

		cpx=coord_px
		cpy=coord_py
		c = 0
		r1, r2 = 0, 0
		a = len(name)
		while r1 < len(cpx):	
			#print(r1)
			xmin = cpx[r1]-20
			xmax = cpx[r1]+20
			ymin = cpy[r1]-20
			ymax = cpy[r1]+20

			a = len(name)
			r2 = 0
			while r2 < a:
				if r1 != r2:
					nn = name[r2]
					pr = prediction[r2]
					con = conclusion[r2]
					x = coord_px[r2]
					y = coord_py[r2]
					cg = coord_geo[r2]
					
					if x>=xmin and x<=xmax and y>=ymin and y<=ymax:
						# delete this index, because this object is detected...
						name.remove(nn)
						prediction.remove(pr)
						conclusion.remove(con)
						coord_px.remove(x)
						coord_py.remove(y)
						coord_geo.remove(cg)
				a = len(name)
				r2+=1
			r1+=1

		df = pd.DataFrame({'imageID': name,
							'predict(0=mamoa, 1=not_mamoa)': prediction,
							'conclusion': conclusion,
							'x': coord_px,
							'y': coord_py,
							'geo': coord_geo})
		df.to_csv(ROOT_DIR+"result_mrcnn_filtered.csv")
		print(ROOT_DIR+"result_mrcnn_filtered.csv --saved")


##############################################################
print("-----------------------------")
print("Detect Archaeological objects")
print("-----------------------------\n\n")
print("Enter the project directory:")
ROOT_DIR = input()

if os.path.exists(ROOT_DIR)==False:
	print(ROOT_DIR, "doesn't exists")
	quit()

print("Options:\n1-YOLO\n2-CNN\n3-MRCNN")
option = int(input())

if option!=1 and option!=2 and option!=3:
	print("Invalid option")
	quit()

print("Insert step (percentage of sliding window)(%):")
step = int(input())
#if step!=20 and step!=50:
#	print("Invalid number")
#	quit()

#geo-coordinates of upper left and lower right corners of LRM image
upper_left = [-8.907277129492169,42.16335245423417]
lower_right = [-8.06051504207768,41.59280385023923]

x_geo1 = upper_left[0]
y_geo1 = upper_left[1]
x_geo2 = lower_right[0]
y_geo2 = lower_right[1]

width = 140000
height = 126000
print("LRM image:\nwidth:", width, "\nheight:", height)

x1 = 0;
y1 = 0;
x2 = width - 1 
y2 = height - 1

#equations to convert geo-coordinates to pixel coordinates
#x = x_geo * m1 + b1
#y = y_geo * m2 + b2

m1 = (x1 - x2)/(x_geo1 - x_geo2)
b1 = x1 - x_geo1 * m1
print('m1:', m1, 'b1:', b1)
m2 = (y1 - y2)/(y_geo1 - y_geo2)
b2 = y1 - y_geo1 * m2
print('m2:', m2, 'b2:', b2)

# geocoordinates of laboreiro region
# TODO -> read these values from text file
geocoordinates_lab = [[-8.443945496150414,41.96002943539938],[-8.444422619963133,41.95940962035295],[-8.445746972976004,41.959075187773934],
[-8.447164967111004,41.95924463361397],[-8.392549897408395,41.8905655591888],[-8.275476199250173,41.97359402080396],
[-8.270892243367232,41.97545346594325],[-8.270575647192437,41.97586816234122],[-8.269228998674299,41.976938346594046],
[-8.262134568898265,41.99370010745395],[-8.261929450249806,41.99408359014454],[-8.317177712302,41.877424588384294],
[-8.485071785166344,41.84396795318029],[-8.418876430027941,41.81337851995365],[-8.300126109873624,41.87799089421808],
[-8.300607692787395,41.8513611427269],[-8.31359259505503,41.87844572252554],[-8.317592408699971,41.88951767044124],
[-8.24084236136806,42.008954692157786],[-8.24111436653232,42.00840622272821],[-8.240793311256473,42.010274586069606],
[-8.240815606761739,42.01068036426547],[-8.333658549795539,41.948498200075164],[-8.388086337253895,41.90367977538698],
[-8.500607293236548,41.94226437680245],[-8.494761411755483,41.82852608623221],[-8.350469360767027,41.837141069467464],
[-8.366062837150928,41.92465092764134],[-8.366062837150928,41.924802537077156],[-8.318849875197046,41.87759403422433],
[-8.289232526000056,41.94531440192301],[-8.455182430805095,41.83820233551818],[-8.313383017305519,41.879426724757295],
[-8.313146684949688,41.8798860121658],[-8.312482278892722,41.88004653980372],[-8.31304858472651,41.880077753511095],
[-8.314310510324633,41.87777685736751],[-8.313168980454952,41.87716596052319],[-8.312602674621164,41.880412186090105],
[-8.283952950352775,41.96563898452462],[-8.284693161127647,41.966071517326796],[-8.290592551821357,41.944440418116535],
[-8.289972736774928,41.94470796417974],[-8.292492128870125,41.9456978846136],[-8.292358355838521,41.94636674977162],
[-8.291662736074183,41.946812659876954],[-8.291252498777267,41.94702223762648],[-8.290142182614957,41.94766434817817],
[-8.288750943086283,41.94671455965379],[-8.288799993197872,41.95161065261047],[-8.319376049121349,41.839165501345725]]

# geocoordinates of Arcos de Valdevez region
# TODO -> read these values from text file
geocoordinates_arcos = [[-8.092653056058893,42.041849480629054],[-8.10168273569212,42.055779712320025],[-8.105950095400269,42.06719501101684],[-8.10588766798552,42.067342161351604],
[-8.11414146403545,42.05331382943747],[-8.09988125886653,42.05490126941249],[-8.135045729774003,42.044373331825305],[-8.101540044458412,42.05582876243161],[-8.090931843052264,42.04254064129235],[-8.112210673279309,42.071903821729286],
[-8.096577064985926,42.07226054981356],[-8.096577064985926,42.07226946801566],[-8.11822154149935,42.07073553725328],[-8.084541951242675,42.04506449248859],[-8.084528573939515,42.04507341069069],[-8.105816322368668,42.06838559099812],
[-8.096492342065911,42.07145791162394],[-8.10310072982712,42.03975370313393],[-8.072043090989865,42.04781129873752],[-8.099533448984358,42.071725457687144],[-8.068743356210314,42.060167467756614],[-8.102592392307026,42.072630655200996],
[-8.081411662303157,42.038429350121056],[-8.099457644266451,42.07189490352718],[-8.096496801166966,42.061803957843225],[-8.108835133781849,42.06357422096144],[-8.096264927912184,42.071979626447195],[-8.139553880939037,42.07392379450649],
[-8.108795001872366,42.05701934241287],[-8.117873731617184,42.07082026017329],[-8.11178705867923,42.063382479616145],[-8.139473617120077,42.07423593158023],[-8.121061988870398,42.06309263804767],[-8.102628065115452,42.05371960763333],
[-8.080800765458834,42.05984195337971],[-8.077888972470934,42.044730059909575],[-8.102324846243818,42.06410039488575],[-8.080979129500973,42.03826436338208],[-8.12393810904987,42.05605171748428],[-8.096046431960566,42.055115306263055],
[-8.12361259467297,42.06290535580343],[-8.119987345516515,42.08031368631608],[-8.095355271297281,42.0714133206134],[-8.089643162847818,42.034781805459346],[-8.064498292007436,42.057104065332894],[-8.110685660719026,42.072153531388274],
[-8.092100127528266,42.073451129794826],[-8.092238359660925,42.061103878977846],[-8.06443586459269,42.05651100489279],[-8.129302407617166,42.062370263677025],[-8.110538510384265,42.07213569498407],[-8.095029756920383,42.07207772667037],
[-8.098365164508357,42.052604832369965],[-8.11042703285793,42.072447832057804],[-8.116736660848554,42.06279833737815],[-8.091796908656631,42.07350017990642],[-8.082628996890747,42.062709155357076],[-8.10105846154464,42.07154709364501],
[-8.095078807031967,42.05591794445268],[-8.089183875439312,42.03549972072894],[-8.116593969614845,42.06265118704338],[-8.129182011888721,42.046192645055115],[-8.119563730916441,42.06983479884048],[-8.097821154179837,42.07158276645343],
[-8.092073372921945,42.03208404932201],[-8.11937198957114,42.069995326478406],[-8.109976663651532,42.06387298073203],[-8.072827892775273,42.06025219067663],[-8.085045829661714,42.07047690939218],[-8.1160811729937,42.06216068592751],
[-8.119019720587918,42.07069540534379],[-8.106659092467767,42.06456414139531],[-8.118917161263688,42.07030746355215],[-8.112701174395184,42.071582766453425],[-8.109517376243026,42.0725147185736],[-8.082053772854854,42.04566647113081],
[-8.082254432402257,42.02836515904344],[-8.078923483915332,42.04319167004614],[-8.131090507139596,42.07129292488496],[-8.1063781691014,42.0680511584191],[-8.103212207353453,42.07321925654004],[-8.115800249627332,42.05645749568014],
[-8.091346539450234,42.03301154234113],[-8.10316761634292,42.063609893769865],[-8.103292471172416,42.054526704924]]

lab_pixel = []
for i in geocoordinates_lab:
	lab_pixel.append(convert_GeoCoord_to_PixelCoord(i, m1, m2, b1, b2))

arcos_pixel = []
for i in geocoordinates_arcos:
	arcos_pixel.append(convert_GeoCoord_to_PixelCoord(i, m1, m2, b1, b2))

print("laboreiro:", len(lab_pixel))
print("Arcos:", len(arcos_pixel))

arcos_px = []
for i in arcos_pixel:
	x = i[0]
	y = i[1]
	cc = (x,y)
	arcos_px.append(cc)

lab_px = []
for i in lab_pixel:
	x = i[0]
	y = i[1]
	cc = (x,y)
	lab_px.append(cc)

d = Detect()

if option == 1:
	d.detectYOLO(ROOT_DIR, step)

if option == 2:
	d.detectCNN(ROOT_DIR, step)

if option == 3:
	d.detectMRCNN(ROOT_DIR, step)

print("Finish!")




