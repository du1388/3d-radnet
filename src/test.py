import os
# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle
import json
import pandas as pd
import numpy as np
from utils.models import RadNet_resnet3d
from utils.processing import CenterImage

### Functions
def format_body_label(labels):
    lab = np.zeros(9, dtype="uint8")
    lab_list = labels.split(';')
    
    for ind in range(len(lab_list)):
        crt = lab_list[ind]
        
        if "B" in crt:
            lab[0] = 1
        if "H" in crt:
            lab[1] = 1
        if "N" in crt:
            lab[2] = 1
        if "Lg" in crt:
            lab[3] = 1
        if "Br" in crt:
            lab[4] = 1
        if "Lv" in crt:
            lab[5] = 1
        if "K" in crt:
            lab[6] = 1
        if "I" in crt:
            lab[7] = 1
        if "P" in crt:
            lab[8] = 1
            
    return lab

def format_view_label(labels):
    lab = np.zeros(3, dtype="uint8")
    
    if "AX" in labels:
        lab[0] = 1
    if "COR" in labels:
        lab[1] = 1
    if "SAG" in labels:
        lab[2] = 1
        
    return lab

def format_seq_label(labels):
    lab = np.zeros(5, dtype="uint8")
    
    if "CT" in labels:
        lab[0] = 1
    if "T1 - SE" in labels:
        lab[1] = 1
    if "T2 - SE" in labels:
        lab[2] = 1
    if "T1 - FLAIR" in labels:
        lab[3] = 1
    if "T2 - FLAIR" in labels:
        lab[4] = 1
    
    return lab

def format_contrast_label(labels):
    lab = np.zeros(2, dtype="uint8")
    lab[labels] = 1
    return lab

### Load configs
with open("./configs/SETTINGS.json") as handle:
    params = json.load(handle)

test_dir = params["TEST_DIR"]
test_label = params["TEST_LABEL"]
output_dir = params["OUTPUT_DIR"]

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def Main():
    # scan lists
    with open(test_label,"rb") as handle:
        test_dict = pickle.load(handle)
        handle.close()

    img_list = list(test_dict.keys())

    # Get Model
    model = RadNet_resnet3d()
    model.load_weights("./models/weights-improvement-21-0.3908-0.2736.hdf5")
    print(model.summary())

    # Get Predictions
    pred_dict = {}
    # Run through scans
    for ind in range(len(img_list)):
        print("Loading scans and running network: {} of {}".format(ind+1,len(img_list)),end="\r")
        crt_path = os.path.join(test_dir,img_list[ind])
        with open(crt_path,"rb") as handle:
            crt_img = pickle.load(handle)
            handle.close()

        # Format image for input
        img_array = crt_img["img_array"]
        img_array = (img_array - np.min(img_array))/(np.max(img_array)-np.min(img_array)).astype("float32")
        img_array = CenterImage(img_array,(48,192,192)) # fix size with zero padding 
        img_array = np.expand_dims(img_array,axis=-1)
        x_test = np.array([img_array])
        pred = model.predict_on_batch(x_test)

        # Parse prediction
        y_seq = format_seq_label(test_dict[img_list[ind]][1])
        y_view = format_view_label(test_dict[img_list[ind]][2])
        y_ctrs = format_contrast_label(test_dict[img_list[ind]][3])
        y_body = format_body_label(test_dict[img_list[ind]][4])
        out_seq = {"prediction":list(pred[0][0]),"label":list(y_seq)}
        out_view = {"prediction":list(pred[1][0]),"label":list(y_view)}
        out_ctrs = {"prediction":list(pred[2][0]),"label":list(y_ctrs)}
        out_body = {"prediction":list(pred[3][0]),"label":list(y_body)}
        pred_dict[img_list[ind]] = [out_seq,out_view,out_ctrs,out_body]

    # Save prediction and labels for analysis later
    with open(os.path.join(output_dir,"test_results_dict"),"wb") as handle:
        pickle.dump(pred_dict,handle)
    print("\nFinished.")

if __name__=="__main__":
    Main()