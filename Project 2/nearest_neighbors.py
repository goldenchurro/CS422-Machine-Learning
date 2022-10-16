
import numpy as np
import math
import scipy.spatial

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    dst=[]
    True_Pos=0
    True_Neg=0
    False_Pos=0
    False_Neg=0
    sum_label=[]
    for j in range (0,len(X_test)):
      for i in range (0,len(X_train)):
        d=scipy.spatial.distance.euclidean([X_train[i]],X_test[j])
        dst.append(d)

    dst=np.array_split(np.array(dst), len(X_test))
    A=np.where(dst[1] == dst[1].min())[0:2]
    A=[]
    for r in range(0,len(X_test)):
      a=dst[r].argsort()[:K]
      A.append(a)
        
    for w in range (0,len(X_test)):
      count=0
      for e in range (0,K):
        s=(Y_train[A[w][e]])
        count=count+s
      if count==0:
        count='take more points'
      elif count<0:
        count=-1
      elif count>0:
        count=1
      sum_label.append(count)

    for u in range (0,len(sum_label)):
      if sum_label[u]==Y_test[u] and sum_label[u]==1:
        True_Pos+=1
      elif sum_label[u]==Y_test[u] and sum_label[u]==-1:
        True_Neg+=1
      elif sum_label[u]!=Y_test[u] and sum_label[u]==1:
        False_Pos+=1
      elif sum_label[u]!=Y_test[u] and sum_label[u]==-1:
        False_Neg+=1

    accuracy=(True_Pos + True_Neg)/float(True_Pos + True_Neg + False_Pos + False_Neg)
    return(accuracy)
