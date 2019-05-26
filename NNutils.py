from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import numpy as np
import os
import pickle
import keras


def find_class_by_fname(df, fname):
    info_str = str.split(fname,'/')[-1]
    info_str = str.split(info_str,'.')
    d,r,t = pd.to_datetime(info_str[1], format='%Y-%m-%d_%H%M%S'), int(info_str[2][2:]), info_str[0][1:]
    return df.loc[d,r]['class']

def show_fits(fname):
    hdulist = fits.open(fname)
    X =  hdulist[0].data
    return X
def constr_filename(date,region, instr_type):
    return 'r'+str(instr_type)+'.'+date.strftime('%Y-%m-%d_%H%M%S')+'.AR'+str(region)+'.fits'

def class_filenames(df, letter, path_to_data, letter_type = 'letter_1'):
    idxs = df[df[letter_type]==letter].index
    return [os.path.join(path_to_data,constr_filename(ind[0],ind[1], df.loc[ind]['instr_type'])) for ind in idxs]
    
def create_fname_dict(df, letter_type):
    names_dict = defaultdict()
    class_ = list(df[letter_type].unique())
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(class_)
    for class_id in class_:
        names_dict[class_id] = class_filenames(df, class_id, letter_type = letter_type)
    return names_dict, label_encoder    

def train_val_split(names_dict, share = 0.2):
    """
    split for each person trials to train and validation
    """
    val_name_dict = {}
    train_name_dict = {}
    for key in names_dict.keys():
        num_fits = share*len(names_dict[key])
        #print(int(np.ceil(num_fits)))
        val_names  = list(np.random.choice(names_dict[key], size = int(np.ceil(num_fits)),replace = False))
        val_name_dict[key] = val_names 
        train_name_dict[key] = list(set(names_dict[key]).difference(set(val_names)))
    return val_name_dict, train_name_dict    

def generate_mcclass_by_name(fnames):
    mcclass = []
    for fname in fnames:
        mcclass.append(find_class_by_fname(df, fname))
    return np.array(mcclass)

def generate_target_by_mcclass(mcclass, letter_ind = 0):
    assert letter_ind in [0,1,2], 'letter_ind should be 0,1 or 2'
    target = np.array([x[letter_ind] for x in mcclass])
    class_ = np.unique(target)
    label_encoder = LabelEncoder()
    label_encoder.fit(class_)
    y = label_encoder.transform(target)
    y = np.squeeze(keras.utils.to_categorical(y, num_classes=len(class_)))

    return y, label_encoder

def load_data_by_name_dict(name_dict, shuffle = True):
    
    fnames = []
    Xlist = []
    ylist = []
    for key in name_dict.keys():
        print('Process: '+ key)
        for fname in tqdm(name_dict[key]):
            hdulist = fits.open(fname)
            X =  hdulist[0].data
            y = label_encoder.transform([key])
            y = np.squeeze(keras.utils.to_categorical(y, num_classes=len(val_name_dict.keys())))
            X = X.reshape(1, X.shape[0], X.shape[1]).astype('float32')
            Xlist.append(X)
            ylist.append(y)
            fnames.append(fname)
    
    print('Merge all data to array')
    y = np.array(ylist)
    fnames = np.array(fnames)    
    x_stats = np.array([log_stats(x) for x in Xlist])
        #x_log = np.array([np.sign(x)*np.log1p(np.abs(x)) for x in batch_X])/8.24
    X= np.array(Xlist)/3900
    if shuffle:
        len_ = y.shape[0]
        idx = np.arange(len_)
        np.random.shuffle(idx)
        return ([x_stats[idx], X[idx]], y[idx], fnames[idx])
    else:
        return ([x_stats, X], y,fnames) 
def log_stats(x):
    stats = np.percentile(x,[0.1, 1,5,10,90,95,99,99.9])
    return np.sign(stats)*np.log1p(np.abs(stats))/8.24