import os
import json
import os.path
from pathlib import Path
import cv2

def createJSON(save, label, coordinates, name, height, width):
	"""

	"""
	# check length of coordinates variable
	shape = []

	for i in coordinates:
		shape.append({
		"label": label,
		"points": i,
		"group_id": None,
		"shape_type": "polygon",
		"flags": {} 	
		})

	dictionary = {
    "version" : "4.5.7",
    "flags" : {},
    "shapes" : [
    	shape
    ],
    "imagePath" : name, 
    "imageHeight" : height,
    "imageWidth" : width
	}
  
	with open(save+"/"+name, "w") as outfile:
	    json.dump(dictionary, outfile, indent=4)

def readYOLOannotations(src, h, w):
	"""
	"""
	with open(src) as f:
		lines = f.readlines()

	values =  []
	for line in lines:
		l = line.strip()
		l = l.replace(' ', ',')
		l = l.split(',')
		values.append(l)

	val = []
	coord = []
	for i in values:
		a=[]
		for j in i:
			a.append(float(j))
		x1 = a[1]*w - (a[3]*w)/2
		y1 = a[2]*h - (a[4]*h)/2
		x2 = a[1]*w + (a[3]*w)/2
		y2 = a[2]*h - (a[4]*h)/2
		x3 = a[1]*w - (a[3]*w)/2
		y3 = a[2]*h + (a[4]*h)/2
		x4 = a[1]*w + (a[3]*w)/2
		y4 = a[2]*h + (a[4]*h)/2
		coord.append([[x1,y1], [x2,y2], [x4,y4], [x3,y3]])
		if i[0] == '0':
			name = "mamoa"
		else:
			name = None
		val.append(a)
	
	return name, coord, h, w


print("Insert path of YOLO files:")
path_src = str(input())
print("Save path:")
save_path = str(input())

if os.path.exists(path_src) == False:
	print("This path",path_src,"doesn't exist")
	quit()
if os.path.exists(save_path) == False:
	os.makedirs(save_path)
	print("Path:", save_path, " created")

# read list of files in source path
files = os.listdir(path_src)

for item in files:
	if item.endswith(".txt"):
		name_file = str(Path(path_src+"/"+item).stem)
		img = cv2.imread(path_src+"/"+name_file+".jpg")
		h = img.shape[0]
		w = img.shape[1]
		name, coord, height, width = readYOLOannotations(path_src+"/"+item, h, w)
		createJSON(save_path, name, coord, name_file+".json", height, width)
	else:
		img = cv2.imread(path_src + "/" + item)
		#print(path_src + "/" + item)
		cv2.imwrite(save_path + "/" + item, img)

print("Dataset YOLO converted...")
#C:/Users/fabri/Desktop/5ano/odyssey-project/preprocessing mamoas/dataset/train/data_15_15
#C:/Users/fabri/Desktop/olaaa