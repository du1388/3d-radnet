import tensorflow as tf
import os


# Functional API
class DenseNet3D():

    def __init__(
        self,
        input_shape,
        output_layers,
        num_channels=1,
        num_blocks=4,
        block_channels = [6,12,24,16],
        initial_filter=64,
        initial_kernal=(3,7,7),
        initial_stride=(1,2,2),
        initial_pool=(2,2,2),
        kernal = (3,3,3),
        growth_rate=32):

            
        self.input_shape = input_shape
        self.output_layers = output_layers

        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.block_channels = block_channels

        self.initial_filter = initial_filter
        self.initial_kernal = initial_kernal
        self.initial_stride = initial_stride
        self.initial_pool = initial_pool
        
        self.kernal = kernal
        self.growth_rate = growth_rate

        self.model_input = tf.keras.Input(shape=[*self.input_shape,self.num_channels], name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        # Iniial Conv
        x = tf.keras.layers.Conv3D(filters=self.initial_filter,kernel_size=self.initial_kernal,strides=self.initial_stride,
         padding="same", name="in_conv/conv")(self.model_input)
        x = tf.keras.layers.BatchNormalization(name="in_conv/bn")(x)
        x = tf.keras.layers.ReLU(name="in_conv/relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=self.initial_pool,padding="valid",name="in_conv/pool")(x)
        
        for ind in range(self.num_blocks):
            x = self.DenseBlock(x, self.block_channels[ind], "Block"+str(ind+1))
            if ind != self.num_blocks-1:
                x = self.TransitionLayer(x, "Trans"+str(ind+1))

        x = tf.keras.layers.BatchNormalization(name="last_/bn")(x)
        x = tf.keras.layers.ReLU(name="last_conv/relu")(x)

        x = tf.keras.layers.GlobalAvgPool3D(name="GlobalPool")(x)
        x = tf.keras.layers.Dense(1000,name="FC1000")(x)
        x = tf.keras.layers.ReLU(name="FC1000/relu")(x)

        x_out_list = []
        for ind in range(len(self.output_layers)):
            x_out = self.output_layers[ind](x)
            x_out_list.append(x_out)

        return x_out_list

    def DenseBlock(self, in_x, num_layers, block_ID):
        layers = []
                
        x = in_x
        layers.append(x)
        layer_pre = block_ID+"_layer"
        for ind in range(num_layers):
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu")(x)
            x = tf.keras.layers.Conv3D(filters=4*self.growth_rate, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/conv_iden")(x)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=self.growth_rate, kernel_size=self.kernal, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv")(x)

            # collect current
            x_c = x

            x = tf.keras.layers.Concatenate(axis=-1,name=layer_pre+str(ind+1)+"/concat")([x]+layers)
            layers.append(x_c)

        return x

    def TransitionLayer(self, x_in, layer_ID):

        in_channels = int(x_in.shape[4]/2)
        layer_pre = layer_ID

        x = tf.keras.layers.BatchNormalization(name=layer_pre+"/bn")(x_in)
        x = tf.keras.layers.ReLU(name=layer_pre+"/relu")(x)
        x = tf.keras.layers.Conv3D(filters=in_channels, kernel_size=1, strides=1, padding="same", name=layer_pre+"/conv")(x)
        x = tf.keras.layers.AveragePooling3D(name=layer_pre+"/pool")(x)

        return x


    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)
    
# Functional API
class ResNet3D():

    def __init__(
        self,
        input_shape,
        output_layers,
        num_channels=1,
        num_blocks=4,
        block_channels = [3,4,6,3],
        initial_filter=64,
        initial_kernal=(3,7,7),
        initial_stride=(1,2,2),
        initial_pool=(2,2,2),
        kernal = (3,3,3),
        mode = 1):

            
        self.input_shape = input_shape
        self.output_layers = output_layers

        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.block_channels = block_channels

        self.initial_filter = initial_filter
        self.initial_kernal = initial_kernal
        self.initial_stride = initial_stride
        self.initial_pool = initial_pool
        
        self.kernal = kernal
        self.mode = mode
        self.model_input = tf.keras.Input(shape=[*self.input_shape,self.num_channels], name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        # Iniial Conv
        x = tf.keras.layers.Conv3D(filters=self.initial_filter,kernel_size=self.initial_kernal,strides=self.initial_stride,
         padding="same", name="in_conv/conv")(self.model_input)
        x = tf.keras.layers.BatchNormalization(name="in_conv/bn")(x)
        x = tf.keras.layers.ReLU(name="in_conv/relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=self.initial_pool,padding="valid",name="in_conv/pool")(x)
        block_list = []
        in_channels = int(x.shape[4])

        for ind in range(self.num_blocks):
            channels = in_channels
            if self.mode == 1:
                x = self.ResBlock(x, self.block_channels[ind], channels,"Block"+str(ind+1))
            else:
                x = self.ResBlock_v1(x, self.block_channels[ind], channels,"Block"+str(ind+1))

            
            if ind != self.num_blocks-1:
                x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name="Block"+str(ind+1)+"pool")(x)
            in_channels = channels*2

        x = tf.keras.layers.BatchNormalization(name="last_/bn")(x)
        x = tf.keras.layers.ReLU(name="last_conv/relu")(x)

        x = tf.keras.layers.GlobalAvgPool3D(name="GlobalPool")(x)
        x = tf.keras.layers.Dense(1000,name="FC1000")(x)
        x = tf.keras.layers.ReLU(name="FC1000/relu")(x)
        
        x_out_list = []
        for ind in range(len(self.output_layers)):
            x_out = self.output_layers[ind](x)
            x_out_list.append(x_out)

        return x_out_list
    
    def ResBlock(self, in_x, num_layers, filters, block_ID):
        layers = []
        channels = filters
        
        x = in_x
        layer_pre = block_ID+"_layer"
        for ind in range(num_layers):
            
            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)
    
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn3")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu3")(x)
            x = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv3")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x
            
        return x

    def ResBlock_v1(self, in_x, num_layers, filters, block_ID):
        layers = []
        channels = filters

        x = in_x
        layer_pre = block_ID+"_layer"
        for ind in range(num_layers):

            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x

        return x

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)

# Functional API
class ResNetV3D():

    def __init__(
        self,
        input_shape,
        output_layers,
        num_channels=1,
        num_blocks=4,
        block_channels = [3,4,6,3],
        initial_filter=64,
        initial_kernal=(3,7,7),
        initial_stride=(1,2,2),
        initial_pool=(2,2,2),
        kernal = (3,3,3),
        mode = 1):

            
        self.input_shape = input_shape
        self.output_layers = output_layers

        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.block_channels = block_channels

        self.initial_filter = initial_filter
        self.initial_kernal = initial_kernal
        self.initial_stride = initial_stride
        self.initial_pool = initial_pool
        
        self.kernal = kernal
        self.mode = mode
        self.model_input = tf.keras.Input(shape=[*self.input_shape,self.num_channels], name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        
        # Iniial Conv
        x = tf.keras.layers.Conv3D(filters=self.initial_filter,kernel_size=self.initial_kernal,strides=self.initial_stride,
         padding="same", name="in_conv/conv")(self.model_input)
        x = tf.keras.layers.BatchNormalization(name="in_conv/bn")(x)
        x = tf.keras.layers.ReLU(name="in_conv/relu")(x)
        x_top = x
        x = tf.keras.layers.MaxPool3D(pool_size=self.initial_pool,padding="valid",name="in_conv/pool")(x)
        
        block_list = []
        in_channels = int(x.shape[4])
        for ind in range(self.num_blocks):
            channels = in_channels
            if self.mode == 1:
                x = self.ResBlock(x, self.block_channels[ind], channels,"Block"+str(ind+1))
            else:
                x = self.ResBlock_v1(x, self.block_channels[ind], channels,"Block"+str(ind+1))

            
            if ind != self.num_blocks-1:
                block_list.append(x)
                x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name="Block"+str(ind+1)+"pool")(x)
                in_channels = channels*2

        x = tf.keras.layers.BatchNormalization(name="last_/bn")(x)
        x = tf.keras.layers.ReLU(name="last_conv/relu")(x)
    
        x = tf.keras.layers.UpSampling3D(size=(2,2,2),name="Latent_Upsampling")(x)
        
        block_list.reverse()
        block_channels = self.block_channels
        block_channels.reverse()
        
        in_channels = in_channels/2
        
        for ind in range(len(block_list)):
            channels = int(in_channels)
            if self.mode == 1:
                x = self.ResBlockUp(x, block_list[ind], block_channels[ind+1], channels ,"Block"+str(ind+5))
            else:
                x = self.ResBlock_v1Up(x, block_list[ind], block_channels[ind+1], channels ,"Block"+str(ind+5))
                
            if ind != self.num_blocks-1:
                block_list.append(x)
                x = tf.keras.layers.UpSampling3D(size=(2,2,2),name="Block"+str(ind+5)+"/Upsampling")(x)
                in_channels = channels/2
        
        x = tf.keras.layers.Concatenate(axis=-1, name="Last_layer/skip_connect")([x,x_top])
        x = tf.keras.layers.Conv3DTranspose(64,(3,3,3),strides=(1,2,2),padding="same", name="Last_layer/Decov")(x)
        x = tf.keras.layers.Conv3D(64,kernel_size=3,strides=1,padding="same",name="Last_layer/conv1")(x)
        x = tf.keras.layers.BatchNormalization(name="Last_layer/bn")(x)
        x = tf.keras.layers.ReLU(name="Last_layer/relu")(x)
        
        x = self.output_layers[0](x)
        return x
    
    def ResBlock(self, in_x, num_layers, filters, block_ID):
        layers = []
        channels = filters
        
        x = in_x
        layer_pre = block_ID+"_layer"
        for ind in range(num_layers):
            
            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)
    
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn3")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu3")(x)
            x = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv3")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x
            
        return x
    
    def ResBlockUp(self, in_x, skip_x, num_layers, filters, block_ID):
        layers = []
        channels = filters
        layer_pre = block_ID+"_layer"
        
        x = tf.keras.layers.Concatenate(axis=-1, name=layer_pre+"/skip_connect")([in_x,skip_x])
        for ind in range(num_layers):
            
            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)
    
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn3")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu3")(x)
            x = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv3")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x
            
        return x

    def ResBlock_v1(self, in_x, num_layers, filters, block_ID):
        layers = []
        channels = filters

        x = in_x
        layer_pre = block_ID+"_layer"
        for ind in range(num_layers):

            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x

        return x
    
    def ResBlock_v1Up(self, in_x, skip_x, num_layers, filters, block_ID):
        layers = []
        channels = filters

        layer_pre = block_ID+"_layer"
        x = tf.keras.layers.Concatenate(axis=-1, name=layer_pre+"/skip_connect")([in_x,skip_x])
        for ind in range(num_layers):

            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x

        return x

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)
    
    
    
class ResNetV3D_noskip():

    def __init__(
        self,
        input_shape,
        output_layers,
        num_channels=1,
        num_blocks=4,
        block_channels = [3,4,6,3],
        initial_filter=64,
        initial_kernal=(3,7,7),
        initial_stride=(1,2,2),
        initial_pool=(2,2,2),
        kernal = (3,3,3),
        mode = 1):

            
        self.input_shape = input_shape
        self.output_layers = output_layers

        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.block_channels = block_channels

        self.initial_filter = initial_filter
        self.initial_kernal = initial_kernal
        self.initial_stride = initial_stride
        self.initial_pool = initial_pool
        
        self.kernal = kernal
        self.mode = mode
        self.model_input = tf.keras.Input(shape=[*self.input_shape,self.num_channels], name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        
        # Iniial Conv
        x = tf.keras.layers.Conv3D(filters=self.initial_filter,kernel_size=self.initial_kernal,strides=self.initial_stride,
         padding="same", name="in_conv/conv")(self.model_input)
        x = tf.keras.layers.BatchNormalization(name="in_conv/bn")(x)
        x = tf.keras.layers.ReLU(name="in_conv/relu")(x)
        x_top = x
        x = tf.keras.layers.MaxPool3D(pool_size=self.initial_pool,padding="valid",name="in_conv/pool")(x)
        
        block_list = []
        in_channels = int(x.shape[4])
        for ind in range(self.num_blocks):
            channels = in_channels
            if self.mode == 1:
                x = self.ResBlock(x, self.block_channels[ind], channels,"Block"+str(ind+1))
            else:
                x = self.ResBlock_v1(x, self.block_channels[ind], channels,"Block"+str(ind+1))

            
            if ind != self.num_blocks-1:
                block_list.append(x)
                x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name="Block"+str(ind+1)+"pool")(x)
                in_channels = channels*2

        x = tf.keras.layers.BatchNormalization(name="last_/bn")(x)
        x = tf.keras.layers.ReLU(name="last_conv/relu")(x)
    
        x = tf.keras.layers.UpSampling3D(size=(2,2,2),name="Latent_Upsampling")(x)
        
        block_list.reverse()
        block_channels = self.block_channels
        block_channels.reverse()
        
        in_channels = in_channels/2
        
        for ind in range(len(block_list)):
            channels = int(in_channels)
            if self.mode == 1:
                x = self.ResBlock(x, block_channels[ind+1], channels ,"Block"+str(ind+5))
            else:
                x = self.ResBlock_v1(x, block_channels[ind+1], channels ,"Block"+str(ind+5))
                
            if ind != self.num_blocks-1:
                block_list.append(x)
                x = tf.keras.layers.UpSampling3D(size=(2,2,2),name="Block"+str(ind+5)+"/Upsampling")(x)
                in_channels = channels/2
        
        x = tf.keras.layers.Conv3DTranspose(64,(3,3,3),strides=(1,2,2),padding="same", name="Last_layer/Decov")(x)
        x = tf.keras.layers.Conv3D(64,kernel_size=3,strides=1,padding="same",name="Last_layer/conv1")(x)
        x = tf.keras.layers.BatchNormalization(name="Last_layer/bn")(x)
        x = tf.keras.layers.ReLU(name="Last_layer/relu")(x)
        
        x = self.output_layers[0](x)
        return x
    
    def ResBlock(self, in_x, num_layers, filters, block_ID):
        layers = []
        channels = filters
        
        x = in_x
        layer_pre = block_ID+"_layer"
        for ind in range(num_layers):
            
            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)
    
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn3")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu3")(x)
            x = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv3")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x
            
        return x
    
    def ResBlockUp(self, in_x, skip_x, num_layers, filters, block_ID):
        layers = []
        channels = filters
        layer_pre = block_ID+"_layer"
        
        x = tf.keras.layers.Concatenate(axis=-1, name=layer_pre+"/skip_connect")([in_x,skip_x])
        for ind in range(num_layers):
            
            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)
    
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)
            
            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn3")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu3")(x)
            x = tf.keras.layers.Conv3D(filters=channels*4, kernel_size=1, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv3")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x
            
        return x

    def ResBlock_v1(self, in_x, num_layers, filters, block_ID):
        layers = []
        channels = filters

        x = in_x
        layer_pre = block_ID+"_layer"
        for ind in range(num_layers):

            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x

        return x
    
    def ResBlock_v1Up(self, in_x, skip_x, num_layers, filters, block_ID):
        layers = []
        channels = filters

        layer_pre = block_ID+"_layer"
        x = tf.keras.layers.Concatenate(axis=-1, name=layer_pre+"/skip_connect")([in_x,skip_x])
        for ind in range(num_layers):

            xi = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/id_bn")(in_x)
            xi = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/id_relu")(xi)
            xi = tf.keras.layers.Conv3D(filters=channels, kernel_size=1, strides=1, padding="same", name=layer_pre+str(ind+1)+"/id_conv")(xi)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn1")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu1")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv1")(x)

            x = tf.keras.layers.BatchNormalization(name=layer_pre+str(ind+1)+"/bn2")(x)
            x = tf.keras.layers.ReLU(name=layer_pre+str(ind+1)+"/relu2")(x)
            x = tf.keras.layers.Conv3D(filters=channels, kernel_size=3, strides=1,
            padding="same", name=layer_pre+str(ind+1)+"/conv2")(x)

            # collect current
            x = tf.keras.layers.Add(name=layer_pre+str(ind+1)+"/add")([xi,x])
            in_x = x

        return x

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)