"""
Created on Tue Apr 17 2018

@author: JQ

Modified on Mon Feb 22 2021

HC Wu

"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D

from numpy import linalg as LA
from sklearn import svm

fcf = glob.glob('data\*.txt') # list .txt files only
file = open(fcf[0], 'r')
text1 = file.read() # string
file.close()

file = open(fcf[50], 'r')
features = file.read().split()
for feature_index in range(len(features)):
    features[feature_index]=features[feature_index]+' '

# Training data
Xtrain = np.zeros((1,np.shape(features)[0])) # empty list
for text_index in range(40):
    file_temp = open(fcf[text_index], 'r')
    text_temp = file_temp.read()
    occurence_temp = []
    for feature_index in features:
        occurence_temp.append(text_temp.count(feature_index)) 
    Xtrain = np.concatenate((Xtrain, [occurence_temp]), axis = 0) # shape: 41-by-13

Xtrain = np.delete(Xtrain, (0), axis = 0) # remove the first empty row 


#XTX = np.dot(Xtrain.T, Xtrain)

  # Subtracting from the mean and scaling 
meanX = np.mean(Xtrain,axis=0)
CtrX = Xtrain-meanX
S1 = np.dot(np.transpose(CtrX),CtrX)
XTX = 1.0/(Xtrain.shape[0]-1)*S1


"""Section 1: Fill in your answers here"""

w,v=LA.eig(XTX)
PC=v[:,0:2]
t=np.dot(CtrX,PC)
# compute eigenvalues and eigenvectors
# Due to the descending order of eigenvalues in w, here we choose the first two columns of v.
# principal comonents; shape: 13-by-2
# PC scores; shape: 40-by-2

t0=t[0:20][:]
t1=t[20:40][:]


fig, ax1  = plt.subplots(1,1, sharey = True)
for i in range(0,20): 
    """ Section 3: Fill in your answers, replace t with the PC scores for class 0 and class 1 respectively"""
    
    class_0 = ax1.scatter(t0[i][0],    t0[i][1],    c = 'r', marker = "o") # the 20 elements for class 0
    class_1 = ax1.scatter(t1[i][0], t1[i][1], c = 'b', marker = "x") # the  20 elements for class 1
    
    """ End of Section 3"""

ax1.set_title('PC scatterplot') 
ax1.set_xlabel('PC2')
ax1.set_ylabel('PC1')
ax1.legend((class_0, class_1), ('CARS', 'HOTELS'), scatterpoints=1, loc='lower left', ncol=1, fontsize=10)

fig.savefig('Q3a_PCscatterplot.png', dpi = 200)   # save the figure to file
plt.close(fig) 

""" End of answer Section 1"""

"""
# ==============  Perceptron =============================
"""
"""Section 2: Fill in your answers here"""
        
X1 =  t0 ## Assign the PC scores for class 1 here
X2 = t1## Assign the PC scores for class 2 here

"""End of section 2"""

Y1 = np.concatenate((np.ones((20,1)),X1), axis=1)
Y2 = np.concatenate((np.ones((20,1))*-1,-X2), axis=1)
Y = np.concatenate((Y1,Y2), axis=0)  # shape: 40-by-3

# Initialize
a = np.zeros((3,1))

# no. of misclassified samples
sum_wrong = 1

a_iter = a
k = 0

while sum_wrong > 0 and k < 1000:
    wrong = np.dot(Y,a_iter) <= 0
    sum_wrong = sum(wrong)
    sum1 = sum(wrong*np.ones((1,3))*Y)
    a_iter = a_iter+sum1.reshape(3,1)
    k=k+1
    
a_con = a_iter

fig, ax1  = plt.subplots(1,1, sharey = True)

x = np.arange(-8,8,1)
y = -(a_con[0]+a_con[1]*x)/a_con[2]

line1 = ax1.plot(x, y, '--', label = "perceptron")

for i in range(0,20): 
    """ Section 3: Fill in your answers, replace t with the PC scores for class 0 and class 1 respectively"""
    
    class_0 = ax1.scatter(X1[i][0],    X1[i][1],    c = 'r', marker = "o") # the 20 elements for class 0
    class_1 = ax1.scatter(X2[i][0], X2[i][1], c = 'b', marker = "x") # the  20 elements for class 1
    
    """ End of Section 3"""

ax1.set_title('Perceptron') 
ax1.set_xlabel('PC2')
ax1.set_ylabel('PC1')
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,5])
ax1.legend((class_0, class_1), ('CARS', 'HOTELS'), scatterpoints=1, loc='lower left', ncol=1, fontsize=10)

fig.savefig('Q4_Perceptron.png', dpi = 200)   # save the figure to file
plt.close(fig) 


# ============== Hard-margin SVM =========================


X = np.concatenate((X1,X2), axis=0)
Y_svm=np.concatenate((np.zeros((1,20)),np.ones((1,20))), axis=1)
Y_svm=Y_svm.ravel()


"""Section 4: Fill in your answers here"""
#Fit Linear SVM
clf=svm.SVC()
clf.fit(X,Y_svm)

"""End of Section 4"""
           
fig, ax1  = plt.subplots(1,1, sharey = True)


for i in range(0,20): 
    class_0 = ax1.scatter(t[i][0],    t[i][1],    c = 'r', marker = "o") # the first  20 elements for class 0
    class_1 = ax1.scatter(t[i+20][0], t[i+20][1], c = 'b', marker = "x") # the latter 20 elements for class 1

# plot the decision function
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax1.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])


ax1.set_title('Hard-margin SVM') 
ax1.set_xlabel('Mapping feature 2')
ax1.set_ylabel('Mapping feature 1')
ax1.legend((class_0, class_1), ('class 0', 'class 1'), scatterpoints=1, loc='lower left', ncol=1, fontsize=10)

fig.savefig('Q4_Hard-margin SVM.png', dpi = 200)   # save the figure to file
fig.show()
plt.close(fig) 
