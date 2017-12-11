"""
Importing necessary libraries

	cv2			-	OpenCV for image proceesing and recognition
	time		- 	for speed comparisons of classifiers/recognizers
	os 			- 	for reading training data directories and path
	numpy		- 	to convertToRGB python lists to numpy arrays as parameter to OpenCV face recognizers
	Flask 		-	getting associated functions needed for routing and rendering
"""

import cv2
import time 
import os
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for
from pymongo import MongoClient
import gridfs


conn = MongoClient("localhost", 27017)
db = conn.gridfs_test
fs = gridfs.GridFS(db)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

haar_face_cascade_frontal = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_alt.xml')
haar_face_cascade_profile = cv2.CascadeClassifier('./classifiers/haarcascade_profileface.xml')
haar_face_cascade = [haar_face_cascade_profile, haar_face_cascade_frontal]

subjects = ["", "Arvind Kejriwal", "Narendra Modi", "None"]


print("Preparing data...")

def allowed_file(filename):
	"""
		purpoose:
			filters file name for only acceptable extenstions that identify an image

		args:
			filename	-	name of the file that will be read

		returns:
			boolean indicating whether the file name and thereby the file is an acceptable image or not
	"""

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
	"""
		purpoose:
			flask function that serve as the first route for the web application

		args:
			None

		returns:
			renders the preliminary home page which is the file upload form for the images
	"""

	if request.method == 'POST':
		if 'file' not in request.files:
			return redirect(request.url)

		file = request.files['file']

		if file.filename == '':
			return redirect(request.url)

		if file and allowed_file(file.filename):
		    return detect_faces_in_image(file)

	return render_template('index.html', data={})    


def detect_faces_in_image(file_stream):
	"""
		purpoose:
			intermediate function that loads the uploaded image and 
			routes to the face recognition function and renders the 
			results returned back from the same

		args:
			file_stream		-	the uploaded iamge file 

		returns:
			renders the resultant web page with the results of the 
			face recognition in addition to the upload form
	"""
	
	img = cv2.imdecode(np.fromstring(file_stream.read(), np.uint8), cv2.IMREAD_COLOR)

	predicted_img, result = predict(img)
	print result
	print("Prediction complete")

	cv2.imwrite('./static/Predicted Image.jpeg',predicted_img)
	return render_template('index.html', data=result)

def convertToRGB(img):
	"""
		purpose:
			re-colors the grayscale image for display

		args:
			img - loaded image
		
		returns:
			colorized image as output
	"""
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor = 1.2):
	"""
		purpoose:
			manipulates a grayscale copy of the image and detects multiscale
			(some images may be closer to camera than others) images and draws
			the faces as rectangles on the original image

		args:
			f_cascade		-	a list containing the classifiers loaded for
								face detection	
			colored_img		-	the uploaded image
			scaleFactor 	-	factor of reduction to deal with multiscale
								images

		returns:
			found_faces 	-	a list containg tuples of the form (img_section, dim)
								where,
									img_section		- grayscale section of the image 
												  where the face has been detected
									dim  			- the co-ordinate dimensions where 
													  the face has been detected

								if no faces are found, it returns the original image as a result

	"""

	img_copy = colored_img.copy()

	gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

	faces = []
	for classifier in f_cascade:
		face = classifier.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
		if face != ():
			faces.append(face)

	# the number of faces found 
	# print('Faces found: ', len(faces))

	if (faces == []):
		return None, None

	found_faces = []

	for face in faces:
		for f in face:
			(x, y, w, h) = f
			cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)	
			found_faces.append((gray[y:y+w, x:x+h], [x, y, w, h]))

	return found_faces

def prepare_training_data(data_folder_path):
	"""
		purpoose:
			this functions detects and learns from faces from all the 
			training images of each person (subject) and returns the 
			faces mapped with the label (the person's identification)

		args:
			data_folder_path 	-	the path to the directory that holds 
									all the training images	

		returns:
			faces 				- 	list of all faces detected in all images
			labels				-	corresponding labels for the respective
									faces so as to tag them

	"""
	 
	dirs = os.listdir(data_folder_path)
	 
	faces = []
	labels = []
	 
	for dir_name in dirs:

		label = int(dir_name)

		subject_dir_path = data_folder_path + "/" + dir_name
		 
		subject_images_names = os.listdir(subject_dir_path)
	 
		for image_name in subject_images_names:
		 	 
			image_path = subject_dir_path + "/" + image_name

			image = cv2.imread(image_path)
			 
			# Display image that is being trained
			# cv2.imshow("Training on image...", image)
			# cv2.waitKey(100)
			 
			found_faces = detect_faces(haar_face_cascade, image)
			 
			if found_faces != (None, None):
				for found_face in found_faces:
					face, rect = found_face
					
					faces.append(face)
					labels.append(label)
			else:
				# print(image_name) # if no face is found in the image
				continue
			 
		# cv2.destroyAllWindows()
		# cv2.waitKey(1)
		
	# cv2.destroyAllWindows()
	 
	return faces, labels

def draw_rectangle(img, rect):
	"""
		purpose:
			draws a rectangle on the image given the (x, y) coordinates,
			the width and the height

		args:
			img 	-	the image on which the rectangle overlay has to
						be drawn
			rect 	-	the dimensions of the detected face, namely the 
						(x, y) coorinates for the corner, the width, and
						the height of the rectangle to be drawn

		returns:
			None

	"""
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 

def draw_text(img, text, x, y):
	"""
		purpose:
			adds the label (name of the person) to the image given the
			(x, y) coordinates

		args:
			img 	-	the image on which the label has to be added
			text 	-	the label of the identified person
			x,y 	- 	the co-ordinates of the starting point for the
						text to be added

		returns:
			None

	"""
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

faces, labels = prepare_training_data("./data/training-data")
print("Data prepared")
 
# Print total faces and labels that are detected and read in the training data
# print("Total faces: ", len(faces))
# print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Alternative recognizers
# face_recognizer = cv2.face.createEigenFaceRecognizer()
# face_recognizer = cv2.face.createFisherFaceRecognizer()

face_recognizer.train(faces, np.array(labels))

def predict(test_img):
	"""
		purpose:
			to leverage the recognizers to detect and identify the people
			whose faces are present in the uploaded image

		args:
			img 	-	the image on which the label has to be added
			text 	-	the label of the identified person
			x,y 	- 	the co-ordinates of the starting point for the
						text to be added

		returns:
			None

	"""
	
	img = test_img.copy()
	found_faces = detect_faces(haar_face_cascade, img)

	face_found = "No"
	is_modi = "No"
	is_kejri = "No"

	if len(found_faces) > 0 and found_faces != (None, None):
		face_found = "Yes"

		for found_face in found_faces:
			face, rect = found_face
	
			label= face_recognizer.predict(face)
	
			label_text = subjects[label[0]]
			 
	
			draw_rectangle(img, rect)
	
			# Add name of the predicted person
			# draw_text(img, label_text, rect[0], rect[1]-5)

			if label[1] > 100:
				continue

			if label_text == "Narendra Modi":
				is_modi = "Yes"

			if label_text == "Arvind Kejriwal":
				is_kejri = "Yes"
		
	result = {
    	"face_found_in_image": face_found,
    	"is_picture_of_modi": is_modi,
    	"is_picture_of_kejri": is_kejri
	}

	return img, result

@app.after_request
def add_header(response):
    """
	    Adding headers to both force latest IE rendering engine or Chrome Frame,
    	and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

"""

	# Manual Testing one image:

	# Detecting a face:

test = cv2.imread('./data/both_1.jpeg')
faces_detected_img = detect_faces(haar_face_cascade, test2)
plt.imshow(convertToRGB(faces_detected_img))
plt.show()
----------------------------------------------------------
	
	# Recognizing a face:


test_img1 = cv2.imread("./data/test-data/both_6.jpeg")
predicted_img1 = predict(test_img1)
print("Prediction complete")
cv2.imshow(subjects[1], predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
