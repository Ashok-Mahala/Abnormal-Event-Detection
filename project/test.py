


from keras.models import load_model
import numpy as np 




def mean_squared_loss(x1,x2):


	''' Compute Euclidean Distance Loss  between 
	input frame and the reconstructed frame'''

	diff=x1-x2
	a,b,c,d,e=diff.shape
	n_samples=a*b*c*d*e
	sq_diff=diff**2
	Sum=sq_diff.sum()
	dist=np.sqrt(Sum)
	mean_dist=((dist/n_samples)*1000)
	print("mean_dist",mean_dist)
	return mean_dist



'''Define threshold for Sensitivity
Lower the Threshhold,higher the chances that a bunch of frames will be flagged as Anomalous.

'''

threshold=0.6


model=load_model('model.h5')

X_test=np.load('testing.npy')
frames=X_test.shape[2]
#Need to make number of frames divisible by 10


flag=0 #Overall video flagq

frames=frames-frames%10

X_test=X_test[:,:,:frames]
X_test=X_test.reshape(-1,227,227,10)
X_test=np.expand_dims(X_test,axis=4)

for number,bunch in enumerate(X_test):
	n_bunch=np.expand_dims(bunch,axis=0)
	reconstructed_bunch=model.predict(n_bunch)

	loss=mean_squared_loss(n_bunch,reconstructed_bunch)
	print(loss)
	if loss>threshold:
		print("Anomalous bunch of frames  {}".format(number))
		flag=1


	else:
		print('Bunch Normal')



if flag==1:
	print("Abnormal Events detected")















