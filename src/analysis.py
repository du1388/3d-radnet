import os
import numpy as np
import pandas as pd
import json
import pickle
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,confusion_matrix, roc_curve
def getMetrics(true_lab, prediction, thres=None):
    auc = roc_auc_score(y_true=true_lab, y_score=prediction)
    
    if thres:
        hlab = np.zeros(len(prediction))
        hlab[np.array(prediction) > thres] = 1
    else:
        hlab = np.round(prediction)
    
    cm = confusion_matrix(y_true=true_lab,y_pred=hlab)
    tn = cm[0,0]
    fn = cm[1,0]
    tp = cm[1,1]
    fp = cm[0,1]

    sen = tp/(tp+fn)
    spec = tn/(tn+fp)
    
    acc = (tp+tn)/(tn+fn+tp+fp)
    
    return auc, acc, sen, spec

def multiclass_performance(y_true,y_pred,label_name=None):
    
    true_lab = np.argmax(np.array(y_true),axis=1)
    pred_lab = np.argmax(np.array(y_pred),axis=1)

    labels = np.arange(0,len(y_true[0]),1,dtype=int)
    cm = confusion_matrix(true_lab,pred_lab,labels=labels)

    tp = 0
    for ind in labels:
        tp = tp + cm[ind,ind]

    total_acc = tp/np.sum(cm)
    
    performance = OrderedDict()
    
    for ind in labels:
        N = np.sum(cm[ind,:])
        
        TP = cm[ind,ind]
        TN = np.sum(cm) - np.sum(cm[ind,:]) - np.sum(cm[:,ind]) + TP
        FP = np.sum(cm[:,ind]) - TP
        FN = np.sum(cm[ind,:]) - TP
        if TP != 0:
            auc = roc_auc_score(np.array(y_true)[:,ind],np.array(y_pred)[:,ind])
            acc = (TP+TN)/(TP+TN+FP+FN)
            sen = TP/(TP+FN)
            spec = TN/(TN+FP)
            ppv = TP/(TP+FP)
            npv = TN/(TN+FN)   
        else:
            auc = 0
            sen = 0
            spec = 0
            ppv = 0
            npv = 0

        if label_name:
            performance[label_name[ind]] = [N,auc,acc,sen,spec,ppv,npv]
        else:
            performance[ind] = [N,auc,acc,sen,spec,ppv,npv]

    return total_acc,performance

def multilabel_performance(y_true,y_pred,label_name=None):
    

    labels = np.arange(0,len(y_true[0]),1,dtype=int)
    performance = OrderedDict()
    total = []
    for ind in range(len(y_true)):
        crt_y_pred = np.array(y_pred)[ind,:]
        crt_y_lab = np.array(y_true)[ind,:]
        crt_y_pred_lab = np.round(crt_y_pred)
        
        if np.array_equal(crt_y_pred_lab,crt_y_lab):
            total.append(1)
        else:
            total.append(0)
            
    total = np.array(total)
    accuray = np.sum(total)/len(total)
    
    for ind in labels:
        crt_y_pred = np.array(y_pred)[:,ind]
        crt_y_lab = np.array(y_true)[:,ind]
        crt_y_pred_lab = np.round(crt_y_pred)
        
        N = np.sum(crt_y_lab)
        if np.sum(crt_y_lab) != 0:
            auc = roc_auc_score(crt_y_lab,crt_y_pred)
            cm = confusion_matrix(crt_y_lab,crt_y_pred_lab,labels=[0,1])
            TP = cm[1,1]
            TN = cm[0,0]
            FP = cm[0,1]
            FN = cm[1,0]
            
            acc = (TP+TN)/(TP+TN+FP+FN)
            sen = TP/(TP+FN)
            spec = TN/(TN+FP)
            ppv = TP/(TP+FP)
            npv = TN/(TN+FN)
        else:
            auc = 0
            sen = 0
            spec = 0
            ppv = 0
            npv = 0

        if label_name:
            performance[label_name[ind]] = [N,auc,acc,sen,spec,ppv,npv]
        else:
            performance[ind] = [N,auc,acc,sen,spec,ppv,npv]
    
    return accuray, performance

### Load configs
with open("./configs/SETTINGS.json") as handle:
    params = json.load(handle)

test_dir = params["TEST_DIR"]
test_label = params["TEST_LABEL"]
output_dir = params["OUTPUT_DIR"]

# label order
columns=["N","AUC","ACC","SEN","SPEC","PPV","NPV"]
seq_label = ["CT","T1 - SE","T2 - SE","T1 - FLAIR", "T2 - FLAIR"]
view_label = ["AX","COR","SAG"]
ctrs_label = ["No Contrast","Contrast"]
body_label = ["Brain","Head","Neck","Lung","Breast","Liver","Kidney","Intestine","Pelvis"]

def Main():
    with open(os.path.join(output_dir,"test_results_dict"),"rb") as handle:
        results_dict = pickle.load(handle)
        handle.close()

    # extract labs
    y_seq = []
    seq_pred = []
    y_view = []
    view_pred = []
    y_ctrs = []
    ctrs_pred = []
    y_body = []
    body_pred = []

    # Load results
    key_list = list(results_dict.keys())
    for ind in range(len(key_list)):
        crt_dict = results_dict[key_list[ind]]
        y_seq.append(crt_dict[0]["label"])
        y_view.append(crt_dict[1]["label"])
        y_ctrs.append(crt_dict[2]["label"])
        y_body.append(crt_dict[3]["label"])
        seq_pred.append(crt_dict[0]["prediction"])
        view_pred.append(crt_dict[1]["prediction"])
        ctrs_pred.append(crt_dict[2]["prediction"])
        body_pred.append(crt_dict[3]["prediction"])

    total_acc, performance = multiclass_performance(y_seq,seq_pred,label_name=seq_label)
    performance["Total Accuracy"] = [len(key_list), None, total_acc, None, None, None, None]
    performance_pd = pd.DataFrame.from_dict(performance,orient='index',columns=columns)
    performance_pd.to_csv(os.path.join(output_dir,"test_results_seq.csv"))
    print(performance_pd)

    total_acc, performance = multiclass_performance(y_view,view_pred,label_name=view_label)
    performance["Total Accuracy"] = [len(key_list), None, total_acc, None, None, None, None]
    performance_pd = pd.DataFrame.from_dict(performance,orient='index',columns=columns)
    performance_pd.to_csv(os.path.join(output_dir,"test_results_view.csv"))
    print(performance_pd)

    total_acc, performance = multiclass_performance(y_ctrs,ctrs_pred,label_name=ctrs_label)
    performance["Total Accuracy"] = [len(key_list), None, total_acc, None, None, None, None]
    performance_pd = pd.DataFrame.from_dict(performance,orient='index',columns=columns)
    performance_pd.to_csv(os.path.join(output_dir,"test_results_ctrs.csv"))
    print(performance_pd)

    total_acc, performance = multilabel_performance(y_body,body_pred,label_name=body_label)
    performance["Total Accuracy"] = [len(key_list), None, total_acc, None, None, None, None]
    performance_pd = pd.DataFrame.from_dict(performance,orient='index',columns=columns)
    performance_pd.to_csv(os.path.join(output_dir,"test_results_body.csv"))
    print(performance_pd)
    
if __name__=="__main__":
    Main()