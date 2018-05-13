from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd

clf = RandomForestClassifier(bootstrap=True,max_depth=10, random_state=10)
# x=np.loadtxt("x_train.txt")
# x=x[:,50:]
# y=np.loadtxt("y_train.txt")

# xval=np.loadtxt("x_val.txt")
# xval=xval[:,50:]
# yval=np.loadtxt("y_val.txt")
# x1=np.vstack([x,xval])
# mean=np.mean(x1,axis=0)
# var=np.var(x1,axis=0)
# print(mean.shape)
# var[var==0]=1
g=open('mask/out_masked_64neg.csv')

y=np.loadtxt("mask/EMP_data.txt")
y=y[:,6]
# print(y)
y=y-11

nl=g.read().split('\n')
x=np.zeros((640,128))


yvalues={0:0,1:1,2:2,3:3 }
for k in range(1,len(nl)):
	arr=nl[k].split(',')
	if len(arr)>63 : 
		censuscode=int(arr[1].rstrip())
		if censuscode>0:
			x[censuscode-1,:]=np.array([float(arr[i]) for i in range(2,len(arr))])
# print(x)
x=np.array(x)
y=np.array(y)
var=np.var(x,axis=0)
# print(var)
for i in range(var.shape[0]):
	if var[i]==0:
		var[i]=1
# var=1
x=(x-np.mean(x))/var

kmeanModel = KMeans(n_clusters=4).fit(x)	
kmeanModel.fit(x)
cluster_labels=kmeanModel.labels_
print(cluster_labels)
mat=np.zeros((4,4))
for i in range(y.shape[0]):
	mat[int(y[i]),int(cluster_labels[i])]+=1
print(mat)

x,xval,y,yval=train_test_split(x,y,test_size=0.2 ,random_state=65)
print('trainy: ',y)


print(x.shape)
print(y.shape)
# x=(x-mean)/np.sqrt(var)
# xval=(xval-mean)/np.sqrt(var)

clf.fit(x, y)
# print(clf.feature_importances_)
ypredict=clf.predict(xval)
y1=clf.predict(x)

err=np.zeros((4,4))
for i in range(xval.shape[0]):
	err[int(ypredict[i]),int(yval[i])]+=1

print(err)

train_acc=np.mean(y1==y)
print("train acc: ",train_acc)
acc=np.mean(ypredict==yval)
print("test acc: ",acc)


from sklearn import svm


clf = svm.SVC( gamma=1,kernel='rbf',C=1e1, decision_function_shape='ovo',degree=5,probability=True)
clf.fit(x,y)
ypredict=clf.predict(xval)
y1=clf.predict(x)
train_acc=np.mean(y1==y)
print("train acc: ",train_acc)
acc=np.mean(ypredict==yval)
err=np.zeros((4,4))
for i in range(xval.shape[0]):
	err[int(ypredict[i]),int(yval[i])]+=1

print(err)


print("test acc: ",acc)


##########################
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e2, hidden_layer_sizes=(10,10), random_state=10,max_iter=5000,learning_rate_init=1e-3,tol=1e-7,validation_fraction=0)
clf.fit(x,y)
ypredict=clf.predict(xval)
y1=clf.predict(x)
train_acc=np.mean(y1==y)
print("train acc: ",train_acc)
acc=np.mean(ypredict==yval)
print("test acc: ",acc)

err=np.zeros((4,4))
for i in range(xval.shape[0]):
	err[int(ypredict[i]),int(yval[i])]+=1

print(err)


