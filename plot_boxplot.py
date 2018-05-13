from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

clf = RandomForestClassifier(max_depth=20, random_state=10)
# x=np.loadtxt("x_train.txt")
# x=x[:,50:]
# y=np.loadtxt("y_train.txt")

# xval=np.loadtxt("x_val.txt")
# xval=xval[:,50:]
# yval=np.loadtxt("y_val.txt")
# x1=np.vstack([x,xval])
# mean=np.mean(x1,axis=0)
# var=np.var(x1,axis=0)

# y=np.loadtxt("NightLight Project/EMP_data.txt")
y=np.loadtxt("mask/EMP_data.txt")	
y=y[:,6]
# y=y[:,1]
g=open('mask/urban_builtup_masked_mean_light_2011.txt')



nl=g.read().split('\n')
x=[0 for i in range(642)]


x1=[]
x2=[]
x3=[]
x4=[]

yvalues={11:0,12:1,13:2,14:3 }
for k in range(1,len(nl)):
	arr=nl[k].split('\t')
	print(arr)
	censuscode=int(arr[4].rstrip())
	# x[censuscode]=[int(arr[i]) for i in range(2,len(arr))]	


	print(arr[5])
	if arr[5]!='':
		x[censuscode]=float(arr[5].rstrip())

	else:
		arr[5]='0'
		x[censuscode]=float(arr[5].rstrip())

	if censuscode!=0:
		if  yvalues[y[censuscode-1]]==0:
			x1.append(float(arr[5]))
		elif yvalues[y[censuscode-1]]==1:
			x2.append(float(arr[5]))
		elif yvalues[y[censuscode-1]]==2:
			x3.append(float(arr[5]))
		elif yvalues[y[censuscode-1]]==3:
			x4.append(float(arr[5]))



data=[x1,x2,x3,x4]
plt.figure()
labelss=['Low employment','Avgerage employment', 'Agricultural employment','Non-Agricultural employment']
plt.xlabel('Employment')
plt.ylabel('Mean NightLight')
plt.title('Employment vs Nighttime light')

plt.boxplot(data,labels=labelss)
plt.show()



print(x)
print(y)

# print(mean.shape)
# var[var==0]=1
# # x=(x-mean)/np.sqrt(var)
# # xval=(xval-mean)/np.sqrt(var)

# clf.fit(x, y)
# # print(clf.feature_importances_)
# ypredict=clf.predict(xval)
# y1=clf.predict(x)

# err=np.zeros((4,4))
# for i in range(xval.shape[0]):
# 	err[int(ypredict[i]),int(yval[i])]+=1

# print(err)

# train_acc=np.mean(y1==y)
# print("train acc: ",train_acc)
# acc=np.mean(ypredict==yval)
# print("test acc: ",acc)


# from sklearn import svm


# clf = svm.SVC( gamma=1,kernel='rbf',C=1e1, decision_function_shape='ovo',degree=5,probability=True)
# clf.fit(x,y)
# ypredict=clf.predict(xval)
# y1=clf.predict(x)
# train_acc=np.mean(y1==y)
# print("train acc: ",train_acc)
# acc=np.mean(ypredict==yval)
# err=np.zeros((4,4))
# for i in range(xval.shape[0]):
# 	err[int(ypredict[i]),int(yval[i])]+=1

# print(err)


# print("test acc: ",acc)


# ##########################
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='adam', alpha=1e2, hidden_layer_sizes=(10,10), random_state=10,max_iter=5000,learning_rate_init=1e-3,tol=1e-7,validation_fraction=0)
# clf.fit(x,y)
# ypredict=clf.predict(xval)
# y1=clf.predict(x)
# train_acc=np.mean(y1==y)
# print("train acc: ",train_acc)
# acc=np.mean(ypredict==yval)
# print("test acc: ",acc)

# err=np.zeros((4,4))
# for i in range(xval.shape[0]):
# 	err[int(ypredict[i]),int(yval[i])]+=1

# print(err)


