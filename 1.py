#----------------------------------------problem#1-1-----------------------------------------------------------
import pylab as plt
import numpy as np

#sigmoid
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

x = plt.linspace(-10,10,100)
plt.plot(x, sigmoid(x), 'b', label='Sig')
plt.grid()
plt.title('Sigmoid Function')
plt.legend(loc='lower right')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

#hingloss
def hingloss(x):
    a= 1-x
    for i in range(len(a)):
        if a[i] < 0:
            a[i] = 0
    return a

x = plt.linspace(-10,10,100)
plt.plot(x, hingloss(x), 'b', label='Hingloss')
plt.grid()
plt.title('Hingloss Function')
plt.legend(loc='lower left')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

#norm1
def norm1(x):
    return abs(x)

x = plt.linspace(-10,10,100)
plt.plot(x, norm1(x), 'b', label='norm 1')
plt.grid()
plt.title('norm1 Function')
plt.legend(loc='lower left')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

#----------------------------------------problem#2-4-----------------------------------------------------------
import numpy as np
from numpy.linalg import inv
import time
import pylab as plt
d=20;
time1= [];
warray = [];

#find time
while d<=500:
    start_time = time.time()
    X = np.random.randn(1000,d)
    y = np.random.randn(1000,1)    
    warray.append(inv((X.T.dot(X))).dot(X.T).dot(y))
    time1.append(time.time()-start_time)
    d = d + 20

#plot
x = plt.linspace(0,500,25)
plt.title('time vs. d')
plt.plot(x, time1, 'b', label='')
plt.xlabel('d')
plt.ylabel('time')
plt.show()

#----------------------------------------problem#2-5-----------------------------------------------------------
import numpy as np
from numpy.linalg import inv
import pylab as plt
from numpy import linalg as LA
#data set
X = np.random.randn(100,40);
y = np.random.randn(100,1);
#hessian matrix
hessian = X.T.dot(X)

#eigenvalue
eigv = LA.eigvals(hessian);
eigv_max = max(eigv)
eigv_min = min(eigv)

#w0 = 0
w1 = np.zeros((40,1));
w2 = np.zeros((40,1));
w3 = np.zeros((40,1));
w4 = np.zeros((40,1));
w5 = np.zeros((40,1));
w6 = np.zeros((40,1));

#w*
w_star=inv((X.T.dot(X))).dot(X.T).dot(y);

# theoretical bound of learning rate
lr = 1/(eigv_max+eigv_min);
lra = np.array([[0.01*lr],[0.1*lr],[lr],[2*lr],[20*lr],[100*lr]]);

w1_list = []
w2_list = []
w3_list = []
w4_list = []
w5_list = []
w6_list = []
w1d = []
w2d = []
w3d = []
w4d = []
w5d = []
w6d = []

#gradient descent with 6 different learning rate
for t in range(100):
    prew = (X.transpose().dot(X.dot(w1)-y))
    w1_list.append(w1)
    w1 = w1-lra[0]*prew

for t in range(100):
    prew = (X.transpose().dot(X.dot(w2)-y))
    w2_list.append(w2)
    w2 = w2-lra[1]*prew
    
for t in range(100):
    prew = (X.transpose().dot(X.dot(w3)-y))
    w3_list.append(w3)
    w3 = w3-lra[2]*prew
    
for t in range(100):
    prew = (X.transpose().dot(X.dot(w4)-y))
    w4_list.append(w4)
    w4 = w4-lra[3]*prew
    
for t in range(100):
    prew = (X.transpose().dot(X.dot(w5)-y))
    w5_list.append(w5)
    w5 = w5-lra[4]*prew
    
for t in range(100):
    prew = (X.transpose().dot(X.dot(w6)-y))
    w6_list.append(w6)
    w6 = w6-lra[5]*prew

#compute ||wt-w*||2
for i in range(100):
    w1d.append(np.linalg.norm(w1_list[i]-w_star))
    w2d.append(np.linalg.norm(w2_list[i]-w_star))
    w3d.append(np.linalg.norm(w3_list[i]-w_star))
    w4d.append(np.linalg.norm(w4_list[i]-w_star))
    w5d.append(np.linalg.norm(w5_list[i]-w_star))
    w6d.append(np.linalg.norm(w6_list[i]-w_star))


#plot
x = plt.linspace(0,100,100)
plt.title('||w^t-w*|| vs. t(0.01lr)')
plt.plot(x, w1d, 'b', label='')
plt.xlabel('Time')
plt.ylabel('||w^t-w*||')
plt.show();

plt.title('||w^t-w*|| vs. t(0.1lr)')
plt.plot(x, w2d, 'b', label='')
plt.xlabel('Time')
plt.ylabel('||w^t-w*||')
plt.show();

plt.title('||w^t-w*|| vs. t(lr)')
plt.plot(x, w3d, 'b', label='')
plt.xlabel('Time')
plt.ylabel('||w^t-w*||')
plt.show();

plt.title('||w^t-w*|| vs. t(2lr)')
plt.plot(x, w4d, 'b', label='')
plt.xlabel('Time')
plt.ylabel('||w^t-w*||')
plt.show();

plt.title('||w^t-w*|| vs. t(20lr)')
plt.plot(x, w5d, 'b', label='')
plt.xlabel('Time')
plt.ylabel('||w^t-w*||')
plt.show();

plt.title('||w^t-w*|| vs. t(100lr)')
plt.plot(x, w6d, 'b', label='')
plt.xlabel('Time')
plt.ylabel('||w^t-w*||')
plt.show();

#----------------------------------------problem#2-6-----------------------------------------------------------
import numpy as np
from numpy.linalg import inv
import pylab as plt
from numpy import linalg as LA

#data set
X = np.random.randn(100,200);
y = np.random.randn(100,1);

#Hessian matrix
hessian = np.dot(X.transpose(),X)

#eigenvalue
eigv = np.linalg.eigvals(hessian);
eigv_max = max(eigv)
eigv_min = min(eigv)

#w^0 = 0
w1 = np.zeros((200,1));
w2 = np.zeros((200,1));
w3 = np.zeros((200,1));
w4 = np.zeros((200,1));
w5 = np.zeros((200,1));
w6 = np.zeros((200,1));

#w*
w_star=inv((X.T.dot(X))).dot(X.T).dot(y);

# theoretical bound of learning rate
lr = 1/(eigv_max+eigv_min);

lra = [0.01*lr,0.1*lr,lr,2*lr,20*lr,100*lr];
w1_list = []
w2_list = []
w3_list = []
w4_list = []
w5_list = []
w6_list = []
w1d = []
w2d = []
w3d = []
w4d = []
w5d = []
w6d = []

##gradient descent with 6 different learning rate to find wt
for t in range(100):
    prew = (X.transpose().dot(X.dot(w1)-y))
    w1_list.append(w1)
    w1 = w1-lra[0]*prew

for t in range(100):
    prew = (X.transpose().dot(X.dot(w2)-y))
    w2_list.append(w2)
    w2 = w2-lra[1]*prew
    
for t in range(100):
    prew = (X.transpose().dot(X.dot(w3)-y))
    w3_list.append(w3)
    w3 = w3-lra[2]*prew
    
for t in range(100):
    prew = (X.transpose().dot(X.dot(w4)-y))
    w4_list.append(w4)
    w4 = w4-lra[3]*prew
    
for t in range(100):
    prew = (X.transpose().dot(X.dot(w5)-y))
    w5_list.append(w5)
    w5 = w5-lra[4]*prew
    
for t in range(100):
    prew = (X.transpose().dot(X.dot(w6)-y))
    w6_list.append(w6)
    w6 = w6-lra[5]*prew

#compute all f(wt)
fw1_list=[];
fw2_list=[];
fw3_list=[];
fw4_list=[];
fw5_list=[];
fw6_list=[];
for i in range(100):
    fw1 = np.linalg.norm(y-X.dot(w1_list[i]))
    fw1 = 0.5*fw1*fw1
    fw1_list.append(fw1)
for i in range(100):
    fw2 = np.linalg.norm(y-X.dot(w2_list[i]))
    fw2 = 0.5*fw2*fw2
    fw2_list.append(fw2)
for i in range(100):
    fw3 = np.linalg.norm(y-X.dot(w3_list[i]))
    fw3 = 0.5*fw3*fw3
    fw3_list.append(fw3)
for i in range(100):
    fw4 = np.linalg.norm(y-X.dot(w4_list[i]))
    fw4 = 0.5*fw4*fw4
    fw4_list.append(fw4)
for i in range(100):
    fw5 = np.linalg.norm(y-X.dot(w5_list[i]))
    fw5 = 0.5*fw5*fw5
    fw5_list.append(fw5)
for i in range(100):
    fw6 = np.linalg.norm(y-X.dot(w6_list[i]))
    fw6 = 0.5*fw6*fw6
    fw6_list.append(fw6)
    
#plot
x = plt.linspace(0,100,100)
plt.title('f(w) vs. t(0.01lr)')
plt.plot(x, fw1_list, 'b', label='')
plt.xlabel('Time')
plt.ylabel('F(w)')
plt.show();

plt.title('f(w) vs. t(0.1lr)')
plt.plot(x, fw2_list, 'b', label='')
plt.xlabel('Time')
plt.ylabel('F(w)')
plt.show();

plt.title('f(w) vs. t(lr)')
plt.plot(x, fw3_list, 'b', label='')
plt.xlabel('Time')
plt.ylabel('F(w)')
plt.show();

plt.title('f(w) vs. t(2lr)')
plt.plot(x, fw4_list, 'b', label='')
plt.xlabel('Time')
plt.ylabel('F(w)')
plt.show();

plt.title('f(w) vs. t(20lr)')
plt.plot(x, fw5_list, 'b', label='')
plt.xlabel('Time')
plt.ylabel('F(w)')
plt.show();

plt.title('f(w) vs. t(100lr)')
plt.plot(x, fw6_list, 'b', label='')
plt.xlabel('Time')
plt.ylabel('F(w)')
plt.show();