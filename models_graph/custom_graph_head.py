import tensorflow
from keras_unet import TF
if TF:
    from tensorflow.keras.models import Model
    import tensorflow as tf
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        MaxPooling2D,
        Dropout,
        SpatialDropout2D,
        UpSampling2D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
        Reshape,
        Flatten,
        Dense,


    )
else:
    from keras.models import Model
    from keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        MaxPooling2D,
        Dropout,
        SpatialDropout2D,
        UpSampling2D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
        Dense,

    )


def custom_adj_unet(input_size= (256, 256, 1), pretrained_weights = None, network_dim = 80, ):
    #= (256, 256, 1)
    inputs = Input(input_size, name = "input_image")
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)


    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

    #state of the art unet ends
    # shrink to node vector


    pool_gh1= MaxPooling2D(pool_size=(2, 2))(conv9)
    unet_out = Dropout(0.5)(pool_gh1)
    conv_gh1 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(unet_out)
    pool_gh2 = MaxPooling2D(pool_size=(2, 2))(conv_gh1)

    #drop_gh3  = Conv2DTranspose(1, 4, output_shape =(unet_out_dimensions[1],unet_out_dimensions[1]) ,activation='relu', padding='same', kernel_initializer='he_normal')(pool_gh2)


    fullyconnected_node_postions = Flatten()(pool_gh2)

    fullyconnected_node_postions = Dense(network_dim*2, name = 'pixel_position_of_nodes')(fullyconnected_node_postions)
    #unet_out.shape[1]/2=network_dim #how it needs to match

    layer_np_1 = fullyconnected_node_postions

    # layer_np_tmp = []
    # for i in range(unet_out_dimensions[1]):
    #     layer_np_tmp.append(Dense(unet_out_dimensions[1],activation='relu')(fullyconnected_node_postions))
    layer_np_tmp = []
    for i in range(unet_out.shape[1]):
        layer_np_tmp.append(layer_np_1)

    layer_np = tensorflow.stack(layer_np_tmp, axis=1)


    #layer_np = layer_np.reshape(layer_np.shape[0], layer_np.shape[1], layer_np.shape[2], 1)
    #layer_np = Reshape((layer_np.shape[0], layer_np.shape[1], layer_np.shape[2], 1))(layer_np)
    layer_np2 = tensorflow.expand_dims(layer_np, axis=-1)

    merge_adj_1 = concatenate([unet_out, layer_np2], axis=-1)
    conv_adj1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_adj_1)
    conv_adj1  = Conv2DTranspose(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_adj1)
    #conv_adj1 = Conv2DTranspose(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_adj1)
    drop_adj1  = Dropout(0.5)(conv_adj1)
    layer_adj_out = MaxPooling2D(pool_size=(6,6))(drop_adj1)
    layer_adj_out = Flatten()(layer_adj_out)
    adj_flatten_dim = int((network_dim * network_dim - network_dim) / 2)
    layer_adj_out = Dense(adj_flatten_dim, name = 'adjacency_matrix')(layer_adj_out)
    # layer_adj_out = Reshape((100,100), name = 'adjacency_matrix')(layer_adj_out)

    model = Model(inputs = inputs, outputs = [fullyconnected_node_postions, layer_adj_out])


    return model


def custom_graph_head(input_shape, max_number_of_nodes = 100, number_of_nodes = 100):
    inputs = Input(input_shape)
    node_positions = Conv2D(2*number_of_nodes, 4, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    merge1 = attention_concat(input_shape, node_positions)
    #merge1 = concatenate(inputs,  node_positions, axis = 3)


    conv2 = Conv2D(256, 256, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv2 = Conv2D(256, 256, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Conv2D(number_of_nodes, number_of_nodes, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv2)

    model = Model(input=inputs, output=conv2)

    return model
