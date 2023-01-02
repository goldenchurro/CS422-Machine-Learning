import numpy as np
from math import log

def RF_build_random_forest(X,Y,max_depth,num_of_trees):
  sample_percent = int(len(X)/10)
  if sample_percent == 0:
    sample_percent+=2
  
  DF = []

  for x in range(num_of_trees):
    index = np.random.randint(len(X), size=sample_percent)
    sub_samples = X[index,:]
    sub_labels = Y[index]
    DT = DT_train_binary(sub_samples,sub_labels,max_depth)
    test_accuracy = DT_test_binary(X,Y,DT)
    print("DT",x,": ",test_accuracy)
    DF.append(DT)
  return DF

def RF_test_random_forest(X,Y,DF):
  correct = 0
  for s_i in range(len(X)):
    out = []
    for dt_i in range(len(DF)):
      dt_predict = predict(X[s_i],DF[dt_i])
      out.append(dt_predict)
    out = np.array(out)
    nonzero_count = np.count_nonzero(out == 1)
    zero_count = np.count_nonzero(out == 0)
    final_prediction = 0
    if nonzero_count > zero_count:
      final_prediction = 1
    if final_prediction == Y[s_i]:
      correct += 1
  return(float(correct)/len(X))

def DT_train_binary(X,Y,max_depth):
  tree = build_tree(X,Y,0,max_depth)
  return tree

def DT_test_binary(X,Y,DT):
  correct = 0
  for s_i in range(len(X)):
    out = predict(X[s_i],DT)
    if out == Y[s_i]:
      correct += 1
  return(float(correct)/len(X))
    
def predict(x,DT):
  out = next(x,DT)
  return out

def next(x,DT):
  if len(DT[1]) == 1:
    if x[DT[0]]< 1:
      return DT[1][0]
  if  len(DT[2]) == 1:
    if x[DT[0]]>0:
      return DT[2][0]
  if x[DT[0]]<1:
    return next(x,DT[1])
  return next(x,DT[2])

def build_tree(X,Y,level,max_depth):
  if sum(Y) == len(Y):
    return [1]
  if sum(Y) == 0:
    return [0]
  #else
  total_samples = len(X)
  num_pos = sum(Y)
  num_neg = total_samples-num_pos
  H = -1*float(num_pos)/total_samples*log(float(num_pos)/total_samples,2)-1*float(num_neg)/total_samples*log(float(num_neg)/total_samples,2)
  tree = []
  IG = []
  for f_i in range(len(X[0])):
    left_Y = []
    right_Y = []
    for s_i in range(len(X)):
      if X[s_i][f_i] > 0:
        right_Y.append(Y[s_i])
      else:
        left_Y.append(Y[s_i])

    #calculate enropy of the split
    if len(left_Y) > 0 and len(right_Y) > 0:
      if sum(left_Y)/len(left_Y) == 0 or sum(left_Y)/len(left_Y) == 1:
        H_left = 0
      else:
        H_left = -1*float(sum(left_Y))/len(left_Y)*log(float(sum(left_Y))/len(left_Y),2)-1*float(len(left_Y)-sum(left_Y))/len(left_Y)*log(float(len(left_Y)-sum(left_Y))/len(left_Y),2)
      if sum(right_Y)/len(right_Y) == 0 or sum(right_Y)/len(right_Y) == 1:
        H_right = 0
      else:
        H_right = -1*float(sum(right_Y))/len(right_Y)*log(float(sum(right_Y))/len(right_Y),2)-1*float(len(right_Y)-sum(right_Y))/len(right_Y)*log(float(len(right_Y)-sum(right_Y))/len(right_Y),2)
      #calculate info gain of the split
      IG.append(H-float(len(left_Y))/(len(left_Y)+len(right_Y))*H_left-float(len(right_Y))/(len(left_Y)+len(right_Y))*H_right)
    else:
      IG.append(0)
  best_ind = IG.index(max(IG))
  left_X = []
  right_X = []
  left_Y = []
  right_Y = []
  for s_i in range(len(X)):
    if X[s_i][best_ind] > 0:
      right_X.append(X[s_i])
      right_Y.append(Y[s_i])
    else:
      left_X.append(X[s_i])
      left_Y.append(Y[s_i])
  if IG[best_ind] == 0 or level==max_depth:
    if sum(Y) > len(Y)-sum(Y):
      return([1])
    else:
      return([0])
  else:
    return([best_ind, build_tree(left_X,left_Y,level+1,max_depth), build_tree(right_X,right_Y,level+1,max_depth)])
  
