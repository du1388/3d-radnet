import os,sys
from tensorflow.keras import layers
from networks import ResNet3D

def RadNet_resnet3d(input_shape=(48,192,192)):

    # Define outputs
    out_seq = layers.Dense(5,activation="softmax",name="out_seq")
    out_view = layers.Dense(3,activation="softmax",name="out_view")
    out_ctrs = layers.Dense(2,activation="softmax",name="out_ctrs")
    out_body = layers.Dense(9,activation="sigmoid",name="out_body")
    out_space = layers.Dense(1,activation="linear",name="out_space")

    NET = ResNet3D(input_shape=input_shape,output_layers=[out_seq,out_view,out_ctrs,out_body,out_space])
    return NET.GetModel()