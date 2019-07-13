#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import keras 
from keras.layers import *
from keras.models import *
from keras.activations import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import os
import itertools
import lex
import yacc
import cpp
import string
import tensorflow as tf
from itertools import product
import math


# In[2]:



folder_name = "/content/drive/My Drive/testing some models/validated/validated/"
# folder_name = "data/"
cols = ["code", "block"]
list_files = [folder_name + i for i in os.listdir(folder_name)]
scanner = lex.lex(cpp)
range_n = 4
lits = cpp.literals
print(lits)
toks = list(cpp.tokens)
print(toks)
toks.remove("CPP_WS")
toks.insert(0, "CPP_WS")
tok2n = dict(zip(toks + [i for i in lits], itertools.count()))
n2tok = dict(zip(itertools.count(), toks + [i for i in lits]))

max_v = 2147483647 - 1

WEIGHTS_FOR_LOSS = np.array([[2,0.5],[0.1,0.1]])

cons_per_line = 10

# df_static = pd.read_excel("suspensionRange.xlsx", header=None)


df1 = pd.read_csv(list_files[0], sep = "`")
df2 = pd.read_csv("/content/drive/My Drive/testing some models/examBefore.csv", sep = "`")
list1 = list(df1.columns)
list2 = list(df2.columns)
ind = -1
for i in range(len(list1)):
    if not (list1[i] == list2[i]):
        ind = i
        break
ind_2 = ind
ind_2 = -1
print(len(lits) + 1)


# In[3]:


print(len(list_files))


# In[4]:


ind_2


# In[5]:


# mean = 0
# l = list_files[:100]
# for i in l : 
#     x = pd.read_csv(i, sep="`")
#     x = x[x.columns[ind:]]
#     x.replace("#empty", np.nan, inplace =True)
#     mean += (x.shape[0] - x.dropna().shape[0])
# print(mean / len(l))


# In[6]:


def get_loss_function(weights, rnn=True):
        
    '''
    gives us the loss function
    '''
    def w_categorical_crossentropy_mine(y_true, y_pred):
        nb_cl = len(weights)
        
        if(not rnn):
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1)
            y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += ( weights[c_t, c_p] * K.cast(y_pred_max_mat, tf.float32)[:, c_p] * K.cast(y_true, tf.float32)[:, c_t]  )
            return K.categorical_crossentropy(y_true, y_pred, True) * final_mask 
        else:
            final_mask = K.zeros_like(y_pred[:, :,0])
            y_pred_max = K.max(y_pred, axis=2)
            y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], K.shape(y_pred)[1], 1))
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += ( weights[c_t, c_p] * K.cast(y_pred_max_mat, tf.float32)[:, :,c_p] * K.cast(y_true, tf.float32)[:, :,c_t]  )
            return K.categorical_crossentropy(y_true, y_pred, True) * final_mask 

            
    return w_categorical_crossentropy_mine


# In[7]:


def get_data(list_files):
    '''
    reads the data and handles the range_n number
    '''
    print("in get data")
    res = []
    for i in list_files:
        try :
            # this part read the files then removes the last col then adds them all together in res
            x = pd.read_csv(i, sep = "`")
            x = x[x.columns[:-1]]
            res.append(x)
        except Exception: 
            print(i)
            continue
    resF = []
    print("after reading")
    for n_i, i in enumerate(res) :
        
        if i.shape[0] is 0 :
            continue
            
        a = i.values
        b = a.copy()
        # first removes the first line from every file then makes a list in which each line(with every thing in front of them) is an element
        b = np.concatenate([b[:, :ind_2], b[:, -1:].astype(np.int) ^ 1, b[:, -1:]], axis = -1)
        
        # this part does sth strange to the b (i think adds "who cares" for block col)
        for j in range(len(b)):
            if np.sum(a[j - range_n : j + range_n, -1]) > 0 :
                b[j, -1] = 1
                b[j, -2] = 0
        for x in range(len(b)):
            for y in range(len(b[x])):
#                 if y < ind  and y > 1 :
                if y > 1 :
                    if type(b[x, y]) == str :
                        try :
                            float(b[x, y].strip())
                        except Exception : 
                            b[x, y] = -3
                elif y == 1 :
                    b[x, y] = "who cares"
        
        # b adds the 0s and 1s layer which i don't know the purpose
        b = pd.DataFrame(b, columns=list(i.columns)[:ind_2] + ["0s", "1s"])
        # replaces all the #empty s with NaN
        b.replace("#empty", np.nan, inplace =True)
        # dropna drops all rows with NaN
        resF.append(b.dropna())
        
        
#     print("this is resF[0]")
#     print(resF[0])
#     print("data was read and changed")
    print("exiting")
    return resF


# In[8]:


def get_replacement(scanner, string_in):
    
    '''
    gets a string and returns the None, 2 which is the tokenized version
    '''
#     print("get_replacement starts")
#     print(string_in)
    scanner.input(string_in)

    # token is the first token of the string_in !
    token = scanner.token()
    
    # same dicts as the starting ones except they don't have the first 10 value and keys!
    id2n = dict(zip([i for i in lits], [tok2n[i] for i in lits]))
    n2id = dict(zip([tok2n[i] for i in lits], [i for i in lits]))
    
    # lits is this: +-*/%|&~^<>=!?()[]{}.,;:\'"
    n_id = len(lits) + 1 # which is 28
    
    res = []
    
    # makes a 2-element list for each token the first element is the tok2n[token] and the second is as follows:
    # 0 for WS, -1 for Strings, -2 for POUND, -3 for DPOUND, -4 for Chars 
    # and id2n[special char] for special characters (if the special char is a new one, we create a space for it which will be the IDs)
    # 
    while token is not None:
#         print("this is the token type")
        t = token.type
#         print(t)
        # tokens are the first 10 values!
        if t in cpp.tokens:
            if token.type == cpp.tokens[cpp.tokens.index("CPP_WS")]:
                #this is because this will make it easier for us to pad our data
                v = 0
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_ID")]:
                v = token.value
#                 print("this is the token value for ID")
#                 print(v)
                if v in id2n.keys():
                    pass
                else :
                    id2n[v] = n_id
                    n2id[n_id] = v
                    n_id += 1
                v = id2n[v]   
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_STRING")]:
                v = -1
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_POUND")]:
                v = -2
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_DPOUND")]:
                v = -3
            elif token.type == cpp.tokens[cpp.tokens.index("CPP_CHAR")]:
                v = -4
            elif token.type in cpp.tokens[3:]:
                print("some thing went really wrong")
            else:
                try :
                    tv = token.value.lower()
                    if tv[-1] == "l" :
                        tv = tv[:-1]
                    if tv[-1] == "u" :
                        tv = tv[:-1]
                    if "x" in  tv :
                        v = int(tv, base = 16)
                    elif tv[-1].lower() == "l":
                        if tv[-2].lower() == "u":
                            v = float(tv[:-2])
                        else :
                            v = float(tv[:-1])
                    else :
                        v = float(tv)
                    v = np.clip(v, - max_v, max_v)
                except Exception as e :
                    print("Couldn't scan this number", token)
                    return
        else:
            # ord returns the ASCII representation
            v = ord(t)
        try:
            t = tok2n[t]
        except Exception:
            n = len(id2n.keys()) + 1 
            tok2n[t] = n
            n2tok[n] = t
            id2n[t] = n
            n2id[n] = t
            t = tok2n[t]
#         print("this is [t, v] to be appended")
#         print([t, v])
        res.append([t, v])
        token = scanner.token()
#         print("----------")
    res = np.array(res)
#     print("get_replacement finishes!")
    return res    


# In[9]:


def tokenize_data(data):
    
    '''
    reads data and tokenizes each of the sentences and adds them together None, NumSen, [NumWord * 2, 1] it also normlaizes data
    '''
    res = []
    x = []
    mean = 0
    max_num = 0
    # i is every excel file
    for i in data:
        if i.shape[0] == 0:
            continue
        temp = []
        mean += i.shape[0]
        max_num = max(max_num, i.shape[0])
        
        # i.values is every line
        for j in i.values:
            try:
                # tok is every 2d array of 2 ints for all the lines and j[0] is every line's actual code
                tok = get_replacement(scanner, j[0]).astype(np.float32)
            except Exception as e:
                continue
            x.append(tok)
            # idk what y is
            y = j[-2:]
            temp.append([tok, y])
        res.append(temp)
        
    mean /= len(res)
    return res, mean, max_num
            
    
    


# In[ ]:





# In[10]:


def change_cols(num, res, empty):
    
    '''
    
    pads or removes data so they all have the same shape in one code  file 
    
    '''
    
    resF = []
    
    for i in res :
        
        temp = []
        
        if (len(i) == 0):
            continue 
            
        for j in i :
            
            if len(j[0]) < num :
                
                result = np.concatenate([j[0], np.ones(( num - len(j[0]), 2 )) * empty], axis = 0)
                
            elif len(j[0]) > num :
                result = j[0][:num, :]
            else :
                result = j[0]
                
            result = result.reshape((-1))
            
            result = np.concatenate([result, np.array([j[1]]).reshape((-1))], axis = 0)
            
            temp.append(np.array(result))
        
        resF.append(np.array(temp))
        
    resF = np.array(resF)
    
    return resF
            
            
            
            
            
            
        
        
        


# In[11]:



def get_final_data(tokenized_final, data):
    
    
    '''
    adds the information from the parser to the things that were gained from the information of scanners
    '''
    
    dataR = np.concatenate([i.drop(cols, axis = 1).values[:, :-2] for i in data], axis = 0)
    dataR = dataR.astype(np.float32)
    
    cnt = 0 
    
    res = []
    
    
    for i in tokenized_final : 
        temp = []
        for j in i :
            
            add = dataR[cnt, :]
            temp.append(np.concatenate([add, j], axis = 0))
            
            cnt += 1
            
            
        res.append(np.array(temp))
    res = np.array(res)
    return res


# In[12]:




def get_acc(y_true, y_pred):
    
    acc = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    rec1 = acc[1][1] / (acc[1][1] + acc[1][0])
    prec1 = acc[1][1] / (acc[1][1] + acc[0][1])
    accuracy = (acc[1][1] + acc[0][0]) / (acc[1][1] + acc[0][0] + acc[1][0] + acc[0][1])
    f1 = 2.0 / ((1.0/rec1) + (1.0/prec1))
    
    return rec1, prec1, accuracy, f1, acc
    
    


# In[13]:


def get_mus(y_true, x, model):
    
    
    y_t = np.argmax(y_true, axis = -1).reshape((-1))
    y_p = model.predict(x)
    y_p = np.argmax(y_p, axis=-1).reshape((-1))
    
    
    
    return get_acc(y_t, y_p)
    
    
    


# In[14]:


def gather_data(list_data, scaler = None, add_all = False, type_add = 0, pad1 = None, pad2 = None, return_before_pad = False):
    
    print("in gather data")
    data = get_data(list_data)
    # tokenize_data returns res(which creates an array of tokens for every line [0, 0] for WS for example) and 
    # mean of the number of lines and maximum number of lines in a file
    r, mean, max_num = tokenize_data(data)
    print("these are mean and max_num" + str(mean) + " " + str(max_num))
#     print("this is a sample from tokenize_data:\n")
#     print(r[0])
#     print("the sample from tokenize_data finished")
    
    # creates a file full of WSs and 0s
    empty = np.array([tok2n["CPP_WS"], 0]).reshape(1, 2).astype(np.float32)
    
    if pad1 is None :
        pad1 = int(mean) + cons_per_line
    
#     print("shape of r before change_cols: ")
#     print(len(r))
    res = change_cols(pad1, r, empty)
#     print("shape after change_cols: ")
#     print(res.shape)
    r = np.array(res)
    
    r = get_final_data(r, data)
#     print("this is r[0] which is after using get_final_data: ")
#     print(r[0])
    
    res = r
    
    if add_all :
        if pad2 is None :
            # calculates the new mean and max from the new res (after get_final_data)
            mean = 0
            max_num = -1
            for i in r :
                mean += i.shape[0]
                max_num = max(max_num, i.shape[0])
            mean /= r.shape[0]
            nums = [int(mean), max_num]
            pad2 = nums[type_add]
#             print("this is pad2 from gather_data: " + str(pad2))
        res = []
        print("reached here 3 ")
        # 
        for i in r :
            if i.shape[0] < pad2 :
                zeros = np.zeros([pad2 - i.shape[0], i.shape[1]])
                zeros[:, -2] = 1
#                 zeros[:, :-2] = scaler.transform(zeros[:, :-2])
                temp = np.concatenate([i, zeros], axis = 0)
            elif i.shape[0] > pad2 :
                temp = i[:pad2, :]
            else :
                temp = i
            res.append(temp)
    res = np.array(res)
    print("this is one example from res in gather_data")
    print(res[0])
    
#     for i, iv in enumerate(res):
#         for j, jv in enumerate(iv):
#             for z, zv in enumerate(jv):
#                 try:
#                     if zv.strip() == "FunctionDecl":
#                         print(i, j, z)
#                         print(iv[j,z])
#                         return
#                 except Exception as e :
# #                     print(e)
#                     continue
    save_r = r.copy()
    
    r = np.concatenate(res, axis = 0).astype(np.float32)
#     print("this is r[0] before scaler in gather_data: ")
#     print(r[0])
    
    if scaler is None :
        scaler = StandardScaler().fit(r[:, :-2].astype(np.float32))
        
    for i, iv in enumerate(res) :
        res[i, :, :-2] = scaler.transform(iv[:, :-2].astype(np.float32)).astype(np.float32)

    if return_before_pad :
        return res, scaler, pad1, pad2, save_r
        
    return res, scaler, pad1, pad2
            
            
                
                
                 
                
    
    
    


# In[57]:


def inception_node(data_in, filters):
    # all the conv layers' strides are (1, 1)
    a = Conv2D(filters, (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(data_in)
    b = Conv2D(filters, (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(data_in)
    b = Conv2D(filters, (3, 3), padding = "same", data_format = "channels_last", activation = 'relu')(b)
    c = Conv2D(filters, (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(data_in)
    c = Conv2D(filters, (5, 5), padding = "same", data_format = "channels_last", activation = 'relu')(c)
    d = MaxPooling2D((3, 3), strides = (1, 1), padding = "same")(data_in)
    d = Conv2D(filters, (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(d)
    
    merged = concatenate([a, b, c, d], axis = 3)
    return merged

def softmax_out(data_in):
    sf = AveragePooling2D((1, 1), strides = (3, 3), data_format = "channels_last")(data_in)
    sf = Conv2D(256, (1, 1), strides = (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(sf)
    sf = Flatten()(sf)
    sf = Dense(1024, activation = "relu")(sf)
    sf = Dropout(0.7)(sf)
    out = Dense(38, activation = "softmax")(sf)
    return out


# In[58]:


def get_model(shape):
    print("in get model")
    
    '''
    gets the first rnn model
    '''
    
#     in1 = Input(shape)
#     print("this is the shape:")
#     print(shape)
#     X = Bidirectional(LSTM(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(in1)
#     X = LSTM(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.1,)(X)
#     X = Dropout(0.2)(X)
#     X = Dense(256, activation=relu)(X)
#     X = Dropout(0.2)(X)
#     X = BatchNormalization()(X)
#     X = Dropout(0.3)(X)
#     X = Dense(128, activation=relu)(X)
#     X = BatchNormalization()(X)
#     X = Dropout(0.25)(X)
#     X = Dense(64, activation=relu)(X)
#     X = BatchNormalization()(X)
#     X = Dropout(0.3)(X)
#     X = Dense(32, activation=relu)(X)
#     X = BatchNormalization()(X)
#     X = Dropout(0.4)(X)
#     X = Dense(16, activation=relu)(X)
#     X = BatchNormalization()(X)
#     X = Dropout(0.4)(X)
#     X = Dense(2, activation=softmax)(X)
#     model = Model(in1, X)
  
    in_images = Input(shape)
    images = Conv2D(64, (7, 7), strides = (2, 2), padding = "same", data_format = "channels_last", activation = "relu")(in_images)
    images = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(images)
    images = BatchNormalization()(images)
    images = Conv2D(192, (1, 1), strides = (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(images)
    images = Conv2D(192, (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(images)
    images = BatchNormalization()(images)
    images = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(images)

    inc_1 = inception_node(images, 256)
    inc_2 = inception_node(inc_1, 480)
    inc_2 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(inc_2)
    inc_3 = inception_node(inc_2, 512)
    sf_out0 = softmax_out(inc_3)

    inc_4 = inception_node(inc_3, 512)
    inc_5 = inception_node(inc_4, 512)
    inc_6 = inception_node(inc_5, 528)
    inc_7 = inception_node(inc_6, 832)
    sf_out1 = softmax_out(inc_6)

    inc_8 = inception_node(inc_7, 832)
    inc_9 = inception_node(inc_8, 1024)
    out = AveragePooling2D((1, 1), data_format = "channels_last")(inc_9)
    out = Dropout(0.4)(out)
    sf_out2 = Flatten()(out)
    sf_out2 = Dense(38, activation = "softmax", name = "sf_out2")(sf_out2)

    model = Model(in_images, [sf_out0, sf_out1, sf_out2])
    
    return model
    


# In[16]:


def max_tokens(data):
    max_res = []
    for i in data:
        temp_max = -1
        i = np.array([i])
        for j in range(i.shape[1]):
            if len(np.amax(np.array(i[:,[j]]))) > temp_max:
                temp_max = len(np.amax(np.array(i[:,[j]])))
        max_res.append(temp_max)
    return max_res


# In[17]:


# def pad_lines(kind, max_lines, data):
    
    


# In[18]:


def pad_tokens(kind, max_tokens, data):
    t = 0
    for i in data:
        i = np.array([i])
        for j in range(i.shape[1]):
            if len(i[0][j]) < max_tokens[t]:
                padd = np.zeros((1, max_tokens[t] - len(i[0][j])))
                data[t][j] = np.append(i[0][j], padd)
            else:
                data[t][j] = np.append(i[0][j], [])
        t += 1
                
    return data


# In[19]:


def get_token_id(data):
    # return the tokens' id from the 2-element arrays (the first element)
    # input shape is (number of files, num_lines, number of tokens per line, 2)
    # output shape is (number of files, num_lines, number of tokens per line)
    res = []
    for i in data:
        i = np.array([i])
        temp_res = []
        for j in range(i.shape[1]):
            temp = i[[0],:,[0]][0][j][:,[0]]
            temp_res.append(temp)
        res.append(temp_res)
    return np.array(res)


# In[20]:


def one_hot(list_files):
    resF = []
    for i in list_files:
        # getting maximum number of tokens
#         for p in tokenize
        
        res = []
        for j in range(tokenize_res.shape[1]):
            onehot = tokenize_res[[0],:,[0]][0][j][:,[0]]
            res.append(onehot)
        res = np.array(res)
        res = np.array([res])
        print(res.shape)
        print(res)
        result = []
        for k in range(res.shape[1]):
            print("this is the array number " + str(k))
            temp = np.array(res[0][k])
            temp = np.array(np.transpose(temp))
            oh = []
            for t in temp[0]:
                tempoh = np.zeros((1, 37))
                tempoh[0][int(t)] = 1
                oh.append(tempoh)
            result.append(oh)
#     res = []
#     for i in range(tokenize_res.shape[1]):
#         onehot = tokenize_res[[0],:,[0]][0][i][:,[0]]
#         res.append(onehot)
#     res = np.array(res)
#     res = np.array([res])
#     print(res.shape)
#     print(res)
#     result = []
#     for i in range(res.shape[1]):
#         print("this is the array number " + str(i))
#         temp = np.array(res[0][i])
#         temp = np.array(np.transpose(temp))
#         oh = []
#         for i in temp[0]:
#             tempoh = np.zeros((1, 37))
#             tempoh[0][int(i)] = 1
#             oh.append(tempoh)
#         result.append(oh)
    return result


# In[60]:


# my code starts

# data = get_data(list_files[:5])
# tokenize_res, mean, max_num = tokenize_data(data)
# tokenize_res = np.array(tokenize_res)
# # print(np.array(tokenize_res).shape)

# tok_res = get_token_id(tokenize_res)
# # tok_res[0][0] = np.append(tok_res[0][0], [[0]], axis = 0)
# max_tok = max_tokens(tok_res)
# print(max_tok)

# print(tok_res[0][0])

# tok_res = pad_tokens("max", max_tok, tok_res)

# print(tok_res)

# # result = one_hot(list_files)

# my code ends

k = 10
l = list_files[:10000]
print("this is the l size " + str(len(l)))
size = math.ceil(len(l) / k)

trs = []
ts = []

for i in range(k):
    
    print("k", i)
    start  = 0
    end    = min(len(l), size)

    data_train = l[:start] + l[end:]
    # print(data_train[0])
    data_test  = l[start : end]
    print("this is the start and end " + str(start) + " " + str(end))
    # print(data_test)
    print("train and test ok")


    r_train, scaler, pad1, pad2 = gather_data(data_train, add_all = True)
    print("first data read")
    r_test, _, _, _ = gather_data(data_test, scaler = scaler, add_all = True, pad1 = pad1, pad2 = pad2)

    print("data read")

    print(r_train.shape[-1])
	
	# RESHAPE HERE
    model = get_model([19, 47, 2])
	
    loss = get_loss_function(WEIGHTS_FOR_LOSS)
#     model.compile(keras.optimizers.adam(lr = 1e-3), keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    #model.summary()
    print("compiled")


    X_train = r_train[:, :, :-2]
    print("this is X_train")
    print(X_train.shape)
	
	# RESHAPE THIS LINE
    X_train = np.array(X_train).reshape(9000, 19, 47, 2)
	
    print("this is a smaple from x_train")
    print(X_train[0])
    Y_train = r_train[:, :, -2:]
    print("this is Y_train")
    print(Y_train.shape)
    Y_train = np.array(Y_train).reshape(9000, 38)
    print("this is a smaple from y_train")
    print(Y_train[0])


    X_test = r_test[:, :, :-2]
	
	# RESHAPE HERE
    X_test = np.array(X_test).reshape(1000, 19, 47, 2)
	
    Y_test = r_test[:, :, -2:]
    Y_test = np.array(Y_test).reshape(1000, 38)

#     model.fit(X_train, y_train, validation_data = [X_test, y_test], epochs = 20, batch_size = 32)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], optimizer = 'sgd', metrics = ['accuracy'])
    model.fit(X_train, [Y_train, Y_train, Y_train], validation_data=(X_test, [Y_test, Y_test, Y_test]), epochs=30, batch_size=64)

    print("train : rec1, prec1, accuracy, f1, acc ")
    trs.append(get_mus(Y_train, X_train,model))
    print(trs[-1])

    print("test : rec1, prec1, accuracy, f1, acc ")
    ts.append(get_mus(Y_test, X_test,model))
    print(ts[-1])


# In[22]:


l = list_files[:]
trs = []
    
    

r_train, scaler, pad1, pad2, save_r = gather_data(l, add_all = True, return_before_pad=True)    
model = get_model([None, r_train.shape[-1] - 2])
loss = get_loss_function(WEIGHTS_FOR_LOSS)
model.compile(keras.optimizers.adam(lr = 1e-3), keras.losses.categorical_crossentropy, metrics = ["accuracy"])


X_train = r_train[:, :, :-2]
Y_train = r_train[:, :, -2:]


model.fit(X_train, Y_train, epochs = 26, verbose = 0, batch_size = 8)

print("train : rec1, prec1, accuracy, f1, acc ")
trs.append(get_mus(Y_train, X_train,model))
print(trs[-1])


# In[16]:


model.save("/content/drive/My Drive/testing some models/model_dyn.h5")


# In[ ]:


pred = model.predict(X_train)


# In[ ]:


pred.shape


# In[17]:


os.makedirs("resesDyn/")
for n_i, i in enumerate(l) : 
    ii = i.index("/")
    i = i[ii+1 : ]
    x = save_r[n_i]
    x = x.reshape((1, ) + x.shape)[:,:,:-2]
    df = pd.DataFrame(model.predict(x).reshape((-1,2)))
    df.to_csv("resesDyn/res_"+i)


# In[37]:


asss = [1,2,3,4,5,6]
print(asss[-2:])


# In[48]:


