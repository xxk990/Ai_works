import scipy.io as sio
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA

#read data from "mnist_01.mat"
data = sio.loadmat("mnist_01.mat")
mat_contents= sio.loadmat("mnist_01.mat")
#the variables in the "mnist_01.mat" file are: 'X_test', 'X_train', 'label_test', 'label_train'



X_test = data['X_test']
X_train = data['X_train']
X_test = X_test.astype('float')
X_train = X_train.astype('float')
label_test = data['label_test']
label_train = data['label_train']

"------------------------------------Display 'X_train' and 'X_test' --------------------------"
chosen_idx = np.random.choice(10000, replace=False, size = 10)
chosen_idx_test = np.random.choice(1000, replace= False, size = 10)
for i in range(len(chosen_idx)):
    if i ==0:
        df = mat_contents['X_train'][chosen_idx[i]]
        df_test = mat_contents['X_test'][chosen_idx_test[i]]
        label_test = mat_contents['label_test'][chosen_idx_test[i]]
        label_train = mat_contents['label_train'][chosen_idx[i]]
    else:
        tem= mat_contents['X_train'][chosen_idx[i]]
        df = np.column_stack((df, tem))
        tem_test= mat_contents['X_test'][chosen_idx_test[i]]
        df_test = np.column_stack((df_test, tem_test))
        lab_test = mat_contents['label_test'][chosen_idx_test[i]]
        label_test = np.column_stack((label_test, lab_test))
        lab_train = mat_contents['label_train'][chosen_idx[i]]
        label_train =np.column_stack((label_train, lab_train))
        
a= []
a_test = []
for l in range(10):
    a.append(df[:,l].reshape(28,28))
    a_test.append(df_test[:,l].reshape(28,28))
    

f, axarr = plt.subplots(5,2)
axarr[0,0].imshow(a[0])
axarr[0,1].imshow(a[1])
axarr[1,0].imshow(a[2])
axarr[1,1].imshow(a[3])
axarr[2,0].imshow(a[4])
axarr[2,1].imshow(a[5])
axarr[3,0].imshow(a[6])
axarr[3,1].imshow(a[7])
axarr[4,0].imshow(a[8])
axarr[4,1].imshow(a[9])

f, axarr = plt.subplots(5,2)
axarr[0,0].imshow(a_test[0])
axarr[0,1].imshow(a_test[1])
axarr[1,0].imshow(a_test[2])
axarr[1,1].imshow(a_test[3])
axarr[2,0].imshow(a_test[4])
axarr[2,1].imshow(a_test[5])
axarr[3,0].imshow(a_test[6])
axarr[3,1].imshow(a_test[7])
axarr[4,0].imshow(a_test[8])
axarr[4,1].imshow(a_test[9])
"------------------------------------Normalize 'X_train' and 'X_test' --------------------------"   

# normalize X_train 
for i in range(len(X_train)):
    X_train[i,:] = X_train[i,:]/LA.norm(X_train[i])
"-----------------------------------------------------------------------------------------------"        
# normalize X_test
for i in range(len(X_test)):
    X_test[i,:] = X_test[i,:]/LA.norm(X_test[i])
"----------------------------------- Generate y_train and y_test--------------------------------"
y_train = data['label_train'].astype('float')
y_test = data['label_test'].astype('float')
y_train[y_train==0] = -1
y_test[y_test<1] = -1
 
"-------------------------------------- Evaluation Metric ---------------------------------------"  
def accor(X_test, y_test, w):
    acc_test = 0
    test_pred = []
    for i in range(len(X_test)):
        test_pred.append(np.sign(np.dot(X_test[i],w)))
    count = 0
    for j in range(len(y_test)):
        if y_test[j] == test_pred[j]:
            count = count + 1
    acc_test=count/len(y_test)
    return acc_test
"-------------------------------------- Convergence of SGD --------------------------------------"

w = np.zeros((784))  #initialize w0 = 0
wt = []
t = []
t2 = []
list1=[]
gradient = np.zeros((784,1)) # Gradient
# SGD 

for j in range(1,20001): #calculate from t = 1 to t = 20000 which means 20000 iterations
    n0 = 1/j #learning rate
    i = np.random.randint(0, 10000) # random choose 
    hingeloss = int(max(1 - y_train[i] * np.dot(X_train[i],w),0)) #function
    

# subgradient of F   

    if  hingeloss > 0:
        gradient = -1 * y_train[i] * X_train[i,:] + w
    else:
        gradient = w
        
    w = w - n0 * gradient   # SGD     
    wt.append(w)
    list1.append(max(1 - y_train[i] * np.dot(X_train[i],w),0)+ 1/2 * (np.linalg.norm(w)) * (np.linalg.norm(w)))
    t.append(j)
    t2.append(1/j)


plt.plot(t,list1)
plt.title('F(wt) v.s. t')
plt.show() 
plt.plot(t2,list1)
plt.title('F(wt) v.s. 1/t')
plt.show()

a = []
sum1= []

for i in range(len(X_test)):
    a.append(np.sign(np.dot(X_test[i],wt[-1])))
"----------------------------------------------------- Hyper-Parameter------------------------------------------------------------------"
ep = [0.000001, 0.001, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 10000]
ep1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
ep_list =[]
ep_list2 =[]
for l in ep:
    w = np.zeros((784))  #initialize w0 = 0
    wt = []
    t = []
    t2 = []
    list1=[]
    gradient = np.zeros((784,1)) # Gradient
        # SGD 
    for j in range(1,10001): #calculate from t = 1 to t = 20000 which means 20000 iterations
        n0 = 1/(j*j) #learning rate
        i = np.random.randint(0, 10000) # random choose 
        hingeloss = int(max(1 - y_train[i] * np.dot(X_train[i],w),0)) #function

        # subgradient of F   
        if  hingeloss > 0:
            gradient = -1 * y_train[i] * X_train[i,:] + l*w
        else:
            gradient = l*w
        
        w = w - n0 * gradient   # SGD     
        wt.append(w)
        t.append(j)
        t2.append(1/j)
    ep_list.append(w)
    ep_list2.append(w)


acc_test = []
for l in ep_list:
    test_pred = []
    for i in range(len(X_test)):
        test_pred.append(np.sign(np.dot(X_test[i],l)))
    count = 0
    for j in range(len(y_test)):
        if y_test[j] == test_pred[j]:
            count = count + 1
    acc_test.append(count/len(y_test))
    

acc_train = []
for l in ep_list:
    train_pred = []
    for i in range(len(X_train)):
        train_pred.append(np.sign(np.dot(X_train[i],l)))
    count = 0
    for j in range(len(y_train)):
        if y_train[j] == train_pred[j]:
            count = count + 1
    acc_train.append(count/len(y_train))

plt.plot(ep1,acc_train)
plt.plot(ep1,acc_test)
plt.title('accuracy v.s. λ')
plt.show() 


#model image
image1 = ep_list[-1].reshape(28,28)
plt.imshow(image1)


ep_list1 = []

for i in ep_list:
    ep_list1.append(LA.norm(i)/len(i))
print(acc_test)
print(ep_list1)
w_star = np.zeros(w.shape)
w_star1 = np.zeros(w.shape)
w_star2 = np.zeros(w.shape)
w_star3 = np.zeros(w.shape)
w_star4 = np.zeros(w.shape)
w_star5 = np.zeros(w.shape)

#hard thresholding
a = abs(w)
s = [10, 20, 50, 100, 200, 400]
w1 = np.argsort(a)[-10:]
for i in range(len(w_star)):
    if i not in w1:
        w_star[i] = 0
    else:
        w_star[i] = w[i]
        

w2 = np.argsort(a)[-20:]
for i in range(len(w_star1)):
    if i not in w2:
        w_star1[i] = 0
    else:
        w_star1[i] = w[i]
        
w3 = np.argsort(a)[-50:]
for i in range(len(w_star2)):
    if i not in w3:
        w_star2[i] = 0
    else:
        w_star2[i] = w[i]

w4 = np.argsort(a)[-100:]
for i in range(len(w_star3)):
    if i not in w4:
        w_star3[i] = 0
    else:
        w_star3[i] = w[i]
        
w5 = np.argsort(a)[-200:]
for i in range(len(w_star4)):
    if i not in w5:
        w_star4[i] = 0
    else:
        w_star4[i] = w[i]
        
w6 = np.argsort(a)[-400:]
for i in range(len(w_star5)):
    if i not in w6:
        w_star5[i] = 0
    else:
        w_star5[i] = w[i]

#model image
image=[]
w_list11 = [w_star, w_star1, w_star2, w_star3, w_star4, w_star5]
for i in w_list11:
    image.append(i.reshape(28,28))
plt.imshow(image[0])
plt.imshow(image[1])
plt.imshow(image[6])

accor =[]

#accuracy
for i in w_list11:
    trainpre =[]
    for l in range(len(X_test)):
        trainpre.append(np.sign(np.dot(X_test[l],i)))
    count = 0
    for j in range(len(y_test)):
        if y_test[j] == trainpre[j]:
            count = count+1
    accor.append(count/len(X_test))
print(accor)
ss = []
for o in range(len(X_test)):
    ss.append(np.sign(np.dot(X_test[o], w_star5)))
    
    
    def acc(X_train,ep_list):
    acc_train = []
    for l in ep_list:
        train_pred = []
        for i in range(len(X_train)):
            train_pred.append(np.sign(np.dot(X_train[i],l)))
        count = 0
        for j in range(len(y_train)):
            if y_train[j] == train_pred[j]:
                count = count + 1
        acc_train.append(count/len(X_train))
    return acc_train

"----------------------------------------noise label-------------------------------------------------------------------------------"

y_train_nose1  = y_train
y_train_nose2  = y_train
y_train_nose3  = y_train
y_train_nose4  = y_train
y_train_nose5  = y_train
y_train_nose6  = y_train
y_train_nose7  = y_train


def random_select(n,list2):
    list3 = list2
    for i in range(n):
        a = np.random.randint(0, 10000)
        if list3[a] == 1:
            list3[a] = -1
        else:
            list3[a] = 1
    return list3
y_train_nose2  = random_select(100, y_train_nose2)
y_train_nose3  = random_select(1000, y_train_nose3)
y_train_nose4  = random_select(2000, y_train_nose4)
y_train_nose5  = random_select(3000, y_train_nose5)
y_train_nose6  = random_select(5000, y_train_nose6)
y_train_nose7  = random_select(7000, y_train_nose7)

def sgd(X_train, y_train):
    ep = [10**-6, 10**-3, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 10000]
    ep1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ep_list =[]
    for l in ep:
        w = np.zeros((784))  #initialize w0 = 0
        wt = []
        t = []
        t2 = []
        gradient = np.zeros((784,1)) # Gradient
        # SGD 

        for j in range(1,10001): #calculate from t = 1 to t = 20000 which means 20000 iterations
            n0 = 1/(j*j) #learning rate
            i = np.random.randint(0, 10000) # random choose 
            hingeloss = int(max(1 - y_train[i] * np.dot(X_train[i],w),0)) #function
    

        # subgradient of F   

            if  hingeloss > 0:
                gradient = -1 * y_train[i] * X_train[i,:] + l*w
            else:
                gradient = l*w
        
            w = w - n0 * gradient   # SGD     
            wt.append(w)
        
            t.append(j)
            t2.append(1/j)
        ep_list.append(w)
    return ep_list
def acc(X_train,ep_list):
    acc_train = []
    for l in ep_list:
        train_pred = []
        for i in range(len(X_train)):
            train_pred.append(np.sign(np.dot(X_train[i],l)))
        count = 0
        for j in range(len(y_train)):
            if y_train[j] == train_pred[j]:
                count = count + 1
        acc_train.append(count/len(X_train))
    return acc_train
y_train_nose1_list = sgd(X_train, y_train_nose1)
y_train_nose1_acc = acc(X_train,y_train_nose1_list)

y_train_nose2_list = sgd(X_train, y_train_nose2)
y_train_nose2_acc = acc(X_train,y_train_nose2_list)

y_train_nose3_list = sgd(X_train, y_train_nose3)
y_train_nose3_acc = acc(X_train,y_train_nose3_list)

y_train_nose4_list = sgd(X_train, y_train_nose4)
y_train_nose4_acc = acc(X_train,y_train_nose4_list)

y_train_nose5_list = sgd(X_train, y_train_nose5)
y_train_nose5_acc = acc(X_train,y_train_nose5_list)

y_train_nose6_list = sgd(X_train, y_train_nose6)
y_train_nose6_acc = acc(X_train,y_train_nose6_list)

y_train_nose7_list = sgd(X_train, y_train_nose7)
y_train_nose7_acc = acc(X_train,y_train_nose7_list)

max_inde = []
max_inde.append(y_train_nose1_acc.index(max(y_train_nose1_acc)))
max_inde.append(y_train_nose2_acc.index(max(y_train_nose2_acc)))
max_inde.append(y_train_nose3_acc.index(max(y_train_nose3_acc)))
max_inde.append(y_train_nose4_acc.index(max(y_train_nose4_acc)))
max_inde.append(y_train_nose5_acc.index(max(y_train_nose5_acc)))
max_inde.append(y_train_nose6_acc.index(max(y_train_nose6_acc)))
max_inde.append(y_train_nose7_acc.index(max(y_train_nose7_acc)))

max_acc = []
max_acc.append(max(y_train_nose1_acc))
max_acc.append(max(y_train_nose2_acc))
max_acc.append(max(y_train_nose3_acc))
max_acc.append(max(y_train_nose4_acc))
max_acc.append(max(y_train_nose5_acc))
max_acc.append(max(y_train_nose6_acc))
max_acc.append(max(y_train_nose7_acc))


def acc1(X_train,ep_list):
    acc_train = []
    for l in ep_list:
        train_pred = []
        for i in range(len(X_train)):
            train_pred.append(np.sign(np.dot(X_train[i],l)))
        count = 0
        for j in range(len(y_test)):
            if y_test[j] == train_pred[j]:
                count = count + 1
        acc_train.append(count/len(X_train))
    return acc_train

max_inde_test=[]
max_acc_test = []
y_test_nose1_acc = acc1(X_test,y_train_nose1_list)

y_test_nose2_acc = acc1(X_test,y_train_nose2_list)

y_test_nose3_acc = acc1(X_test,y_train_nose3_list)

y_test_nose4_acc = acc1(X_test,y_train_nose4_list)

y_test_nose5_acc = acc1(X_test,y_train_nose5_list)

y_test_nose6_acc = acc1(X_test,y_train_nose6_list)

y_test_nose7_acc = acc1(X_test,y_train_nose7_list)

max_inde_test.append(y_test_nose1_acc.index(max(y_test_nose1_acc)))
max_inde_test.append(y_test_nose2_acc.index(max(y_test_nose2_acc)))
max_inde_test.append(y_test_nose3_acc.index(max(y_test_nose3_acc)))
max_inde_test.append(y_test_nose4_acc.index(max(y_test_nose4_acc)))
max_inde_test.append(y_test_nose5_acc.index(max(y_test_nose5_acc)))
max_inde_test.append(y_test_nose6_acc.index(max(y_test_nose6_acc)))
max_inde_test.append(y_test_nose7_acc.index(max(y_test_nose7_acc)))

max_acc_test.append(max(y_test_nose1_acc))
max_acc_test.append(max(y_test_nose2_acc))
max_acc_test.append(max(y_test_nose3_acc))
max_acc_test.append(max(y_test_nose4_acc))
max_acc_test.append(max(y_test_nose5_acc))
max_acc_test.append(max(y_test_nose6_acc))
max_acc_test.append(max(y_test_nose7_acc))

ep_train = []
for i in max_inde:
    ep_train.append(ep[i])
ep_test = []
for i in max_inde_test:
    ep_test.append(ep[i])

norm_w = []
norm_w.append(LA.norm(y_train_nose1_list)/len(y_train_nose1_list))
norm_w.append(LA.norm(y_train_nose2_list)/len(y_train_nose2_list))
norm_w.append(LA.norm(y_train_nose3_list)/len(y_train_nose3_list))
norm_w.append(LA.norm(y_train_nose4_list)/len(y_train_nose4_list))
norm_w.append(LA.norm(y_train_nose5_list)/len(y_train_nose5_list))
norm_w.append(LA.norm(y_train_nose6_list)/len(y_train_nose6_list))
norm_w.append(LA.norm(y_train_nose7_list)/len(y_train_nose7_list))
#best  Hyper-Parameter for train set
print('Hyper-Parameter for train set:',ep_train)
#best accuracy training set
print('accuracy training set:', max_acc)
#best  Hyper-Parameter for test set
print('Hyper-Parameter for test set:', ep_test)
#best accuracy test set
print('accuracy test set:',max_acc_test)
#norm /d
print('norm /d:',norm_w)