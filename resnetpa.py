"""
Resnetpa:
reference: He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
forked from: https://gist.github.com/JefferyRPrice/c1ecc3d67068c8d9b3120475baba1d7e
"""

from keras.models import Model
from keras.layers import Input, merge, add
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D
from keras.regularizers import l2
from keras import backend as K



def rnpa_bottleneck_layer(input_tensor, nb_filters, filter_sz, stage, init='glorot_normal', reg=0.0,
                          use_shortcuts=True):
    nb_in_filters, nb_bottleneck_filters = nb_filters

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = 'add' + str(stage)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage > 1:  # first activation is just after conv1
        x = BatchNormalization(axis=-1, name=bn_name + 'a')(input_tensor)
        x = Activation('relu', name=relu_name + 'a')(x)
    else:
        x = input_tensor

    x = Conv2D(
        filters=nb_bottleneck_filters,
        kernel_size=(1, 1),
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name=conv_name + 'a'
    )(x)

    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = BatchNormalization(axis=-1, name=bn_name + 'b')(x)
    x = Activation('relu', name=relu_name + 'b')(x)
    x = Conv2D(
        filters=nb_bottleneck_filters,
        kernel_size=(filter_sz, filter_sz),
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name=conv_name + 'b'
    )(x)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = BatchNormalization(axis=-1, name=bn_name + 'c')(x)
    x = Activation('relu', name=relu_name + 'c')(x)
    x = Conv2D(
        filters=nb_in_filters,
        kernel_size=(1, 1),
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        name=conv_name + 'c'
    )(x)

    # merge
    if use_shortcuts:
        x = add([x, input_tensor], name=merge_name)
    return x


def ResNetPreAct(input_shape=(32, 32, 3), nb_classes=10, layer1_params=(5, 64, 2), res_layer_params=(3, 16, 3),
                 final_layer_params=None, init='glorot_normal', reg=0.0, use_shortcuts=True):
    sz_L1_filters, nb_L1_filters, stride_L1 = layer1_params
    sz_res_filters, nb_res_filters, nb_res_stages = res_layer_params

    use_final_conv = (final_layer_params is not None)
    if use_final_conv:
        sz_fin_filters, nb_fin_filters, stride_fin = final_layer_params
        sz_pool_fin = input_shape[1] / (stride_L1 * stride_fin)
    else:
        sz_pool_fin = input_shape[1] / (stride_L1)

    # Permute dimension order if necessary
    # if K.image_data_format() == 'channels_last':
    #    input_shape = (input_shape[1], input_shape[2], input_shape[0])

    img_input = Input(shape=input_shape, name='input')

    x = Conv2D(
        filters=nb_L1_filters,
        kernel_size=(sz_L1_filters, sz_L1_filters),
        padding='same',
        strides=(stride_L1, stride_L1),
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name='conv0'
    )(img_input)

    x = BatchNormalization(axis=-1, name='bn0')(x)
    x = Activation('relu', name='relu0')(x)

    for stage in range(1, nb_res_stages + 1):
        x = rnpa_bottleneck_layer(
            x,
            (nb_L1_filters, nb_res_filters),
            sz_res_filters,
            stage,
            init=init,
            reg=reg,
            use_shortcuts=use_shortcuts
        )

    x = BatchNormalization(axis=-1, name='bnF')(x)
    x = Activation('relu', name='reluF')(x)

    if use_final_conv:
        x = Conv2D(
            filters=nb_L1_filters,
            kernel_size=(sz_L1_filters, sz_L1_filters),
            padding='same',
            strides=(stride_fin, stride_fin),
            kernel_initializer=init,
            kernel_regularizer=l2(reg),
            name='convF'
        )(x)
    print(x.shape)
    x = AveragePooling2D((sz_pool_fin, sz_pool_fin), name='avg_pool')(x)

    # x = Flatten(name='flat')(x)
    x = Flatten()(x)
    # x = Dense(nb_classes, activation='softmax', name='fc10')(x)
    x = Dense(nb_classes, activation=None, name='fc10')(x)

    return Model(img_input, x, name='rnpa')
