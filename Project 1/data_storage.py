import numpy as np

def build_nparray(file_data):
  file_data = np.delete(file_data, (0), axis=0)
  X = []
  Y = []
  for row in file_data:
    temp = [float(x) for x in row]
    temp.pop(-1)
    X.append(temp)
    Y.append(int(row[-1]))
  X = np.array(X)
  Y = np.array(Y)
  return X, Y

def build_list(file_data):
  file_data = np.delete(file_data, (0), axis=0)
  X = []
  Y = []
  for row in file_data:
    temp = [float(x) for x in row]
    temp.pop(-1)
    X.append(temp)
    Y.append(int(row[-1]))
  return X,Y

def build_dict(file_data):
  headers = file_data[0]
  file_data = np.delete(file_data, (0), axis=0)
  X = []
  Y = []
  for row in file_data:
    temp = [float(x) for x in row]
    temp.pop(-1)
    X.append(temp)
    Y.append(int(row[-1]))
  X_dict = {}
  Y_dict = {}
  index_list = []
  for index in range(len(X)):
    X_dict_line = dict(zip(headers, X[index]))
    Y_dict_line = {index, Y[index]}
    X_dict[index] = X_dict_line
    index_list.append(index)
  Y_dict = dict(zip(index_list, Y))
  return X_dict, Y_dict
