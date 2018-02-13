import csv
import numpy as np
import cv2
import keras
import sys
import sklearn
from random import randrange
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model
from keras import optimizers
from keras import backend as K
K.set_image_dim_ordering('tf')


#flags 
captured_data=[]
straight_data=[]
turn_data=[]
saved_model=''
#training data folder num
index_data=0
#decay rate for learning
decay=0.1
#resume training on older model ?
use_prev=0
count=0
steering=[]
images=[]
data_redis=False
keep_prob=[]
# command to use : python model_nvidia.py <training_data_folder_num> <normalise_data_distribution> <previous_model_name>
# python model_nvidia.py 0 t old_model.h5

#parse inputs
if len(sys.argv)>1:
	index_data=int(sys.argv[1])
	if (len(sys.argv) >= 4):
		saved_model=str(sys.argv[3])	
		use_prev=1;
	if (len(sys.argv) >=3):
		# if data redistribution flag is on then we even the input data based on the steering steering
		if sys.argv[2]=='t':
			data_redis=True
		else:
			data_redis=False
	with open('data/'+str(index_data)+'/driving_log.csv') as csvfile:
		reader=csv.reader(csvfile)
		print(csvfile)
		for line in reader:
			# if header or the speed is too low ,ignore
			if line[3]=='steering'or np.abs(float(line[6]))<1:
				continue
			steering.append(float(line[3]))
			images.append(line)

# code to evenly distribute input data.
if data_redis==True:
	#placing data into bins of  a histogram
	nbins=25
	remove_list=[]
	avg_samples=len(steering)/nbins
	hist,bins=np.histogram(steering,nbins)

	target=avg_samples*.5
	#deciding to keep or not keep the data
	for i in range(nbins):
	    if hist[i]<target:
	        keep_prob.append(1)
	    else:
	        keep_prob.append(1./(hist[i]/target))

	# randomly select indexes to remove
	for i in range(len(steering)):
	    for j  in range(nbins):
	        if steering[i]>bins[j] and steering[i]<=bins[j+1]:
	            if np.random.rand() >keep_prob[j]:
	                remove_list.append(i)
	# delete selected indexes
	for i in sorted(remove_list,reverse=True):
	    del steering[i]
	    del images[i]

captured_data=images
print ("test data.printing one line")
print(captured_data[2])
print("number of samples:",len(captured_data))

sklearn.utils.shuffle(captured_data)

train_data,validation_data=train_test_split(captured_data,test_size=0.2)
def gen_model():
	model = Sequential()
	model.add(Lambda(lambda x :(x/255.0)-0.5,input_shape=input_shape))
	#keras version is 1.2.x
	model.add(Conv2D(24,5,5,activation='elu'))
	model.add(Conv2D(36,5,5,activation='elu'))
	model.add(Conv2D(48,5,5,activation='elu'))
	model.add(Dropout(0.2))
	model.add(Conv2D(64,3,3,activation='elu'))
	model.add(Dropout(0.2))
	model.add(Conv2D(64,3,3,activation='elu'))
	model.add(Flatten())
	#higher dropout rate in the last layer .Earlier layers captures most of the features.
	model.add(Dropout(0.4))
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

def generator(samples,batch_size=128):

	num_samples=len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0,num_samples,batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images=[]
			measurements=[]
			for batch_sample in batch_samples:
				source_path=batch_sample[0]
				# need to parse driving_log.csv differently if data was collected on a windows system
				if len(source_path.split('/'))==1:
					#windows
					filename=source_path.split('\\')[-3]+'/'+source_path.split('\\')[-2]+'/'+source_path.split('\\')[-1]
				else:
					filename=source_path.split('/')[-3]+'/'+source_path.split('/')[-2]+'/'+source_path.split('/')[-1]
				corrections=[0.0,0.28,-0.28]
				replacement_strings=['center','left','right']
				# the file names for left,center and right differ only by the strings "left","right" and "center".
				# runnign the loop 3 times (once each for left,right and center)
				for replacement_string,correction in zip(replacement_strings,corrections):
					current_path='data/'+filename.replace('center',replacement_string)
					try:
						#prerprocess data
						image=cv2.imread(current_path)
						#cropping 
						image=image[65:65+70,:,:]
						#resizing to input size expected by the model
						image=cv2.resize(image,(200,66))
						# Converting to YUV as mentioned in the nvidia paper
						image=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
						#adding some blur to reduce noise
						image=cv2.GaussianBlur(image,(3,3),0)
					except Exception as e:
						print("Exiting",e)
						print(current_path)
						exit()
					#append the image data measurement data 
					images.append(image)
					measurement=float(line[3])+correction
					measurements.append(measurement)
					# add flipped image for each image
					images.append(np.fliplr(image))
					measurements.append(-measurement)

				X_train=np.array(images)
				y_train=np.array(measurements)
				yield sklearn.utils.shuffle(X_train,y_train)

train_generator=generator(train_data,batch_size=4)
validation_generator=generator(validation_data,batch_size=4)

input_shape=(66,200,3)

if use_prev:
	model=load_model(saved_model)
	# if using a previous model.Multiply the learing rate by 'decay' to reduce the learning rate
	K.set_value(model.optimizer.lr,decay*(K.get_value(model.optimizer.lr)))
	print("new lr=",K.get_value(model.optimizer.lr))
else:
	#new model
	model = gen_model()
	adam=optimizers.Adam(decay=0.00001,lr=0.0001)
	model.compile(loss='mse',optimizer=adam)
	print("learning rate=",K.get_value(model.optimizer.lr))
#fit the data
model.fit_generator(train_generator,validation_data=validation_generator,nb_epoch=7,nb_val_samples=len(validation_data),samples_per_epoch=len(train_data))
#save the model
model.save('nvidia_redis_'+str(index_data)+'.h5')

