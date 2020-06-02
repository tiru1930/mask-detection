import tensorflow as tf
from config import config


class slimModel(object):
    """docstring for simModel"""
    def __init__(self, num_of_cells):
        super(slimModel, self).__init__()
        self.num_of_cells = num_of_cells

        self.image_sizes  = config.imgae_dim
        self.base_channel = config.base_channel
        self.num_of_classes = len(config.class_names)


    def _conv_block(self,inputs, filters, kernel=(3, 3), 
                    strides=(1, 1), use_bn=True, padding=None, block_id=None):

        if block_id is None:
            block_id = (tf.keras.backend.get_uid())

        if strides == (2, 2):
            x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv_pad_%d' % block_id)(inputs)
            x = tf.keras.layers.Conv2D(filters, kernel,
                                       padding='valid',
                                       use_bias=False if use_bn else True,
                                       strides=strides,
                                       name='conv_%d' % block_id)(x)
        else:
            x = tf.keras.layers.Conv2D(filters, kernel,
                                       padding='same',
                                       use_bias=False if use_bn else True,
                                       strides=strides,
                                       name='conv_%d' % block_id)(inputs)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name='conv_bn_%d' % block_id)(x)

        return tf.keras.layers.ReLU(name='conv_relu_%d' % block_id)(x)

    def _branch_block(self,input, filters):

        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(input)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
        x1 = tf.keras.layers.Conv2D(filters * 2, kernel_size=(3, 3), padding='same')(input)
        x = tf.keras.layers.Concatenate(axis=-1)([x, x1])
        return tf.keras.layers.ReLU()(x)

    def _depthwise_conv_block(self,inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), use_bn=True, block_id=None):

        if block_id is None:
            block_id = (tf.keras.backend.get_uid())

        if strides == (1, 1):
            x = inputs
        else:
            x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name='conv_pad_%d' % block_id)(inputs)

        x = tf.keras.layers.DepthwiseConv2D((3, 3),
                                            padding='same' if strides == (1, 1) else 'valid',
                                            depth_multiplier=depth_multiplier,
                                            strides=strides,
                                            use_bias=False if use_bn else True,
                                            name='conv_dw_%d' % block_id)(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)

        x = tf.keras.layers.ReLU(name='conv_dw_%d_relu' % block_id)(x)

        x = tf.keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                                   padding='same',
                                   use_bias=False if use_bn else True,
                                   strides=(1, 1),
                                   name='conv_pw_%d' % block_id)(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)

        return tf.keras.layers.ReLU(name='conv_pw_%d_relu' % block_id)(x)

    def _create_head_block(self,inputs, filters, strides=(1, 1), block_id=None):

        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(inputs)
        return x

    def _compute_heads(self,x, idx, num_class, num_cell):

        conf = self._create_head_block(inputs=x, filters=num_cell[idx] * num_class)
        conf = tf.keras.layers.Reshape((-1, num_class))(conf)
        loc = self._create_head_block(inputs=x, filters=num_cell[idx] * 4)
        loc = tf.keras.layers.Reshape((-1, 4))(loc)

        return conf, loc

    def build_model(self):
        
        inputs = tf.keras.layers.Input(shape=[self.image_sizes[0],self.image_sizes[1], 3], name='input_image')
        x = self._conv_block(inputs, self.base_channel, strides=(2, 2))  
        x = self._conv_block(x, self.base_channel * 2, strides=(1, 1))
        x = self._conv_block(x, self.base_channel * 2, strides=(2, 2))  
        x = self._conv_block(x, self.base_channel * 2, strides=(1, 1))
        x = self._conv_block(x, self.base_channel * 4, strides=(2, 2)) 
        x = self._conv_block(x, self.base_channel * 4, strides=(1, 1))
        x = self._conv_block(x, self.base_channel * 4, strides=(1, 1))
        x = self._conv_block(x, self.base_channel * 4, strides=(1, 1))
    
        x1 = self._branch_block(x, self.base_channel)

        x = self._conv_block(x, self.base_channel * 8, strides=(2, 2))  
        x = self._conv_block(x, self.base_channel * 8, strides=(1, 1))
        x = self._conv_block(x, self.base_channel * 8, strides=(1, 1))

        x2 = self._branch_block(x, self.base_channel)

        x = self._depthwise_conv_block(x, self.base_channel * 16, strides=(2, 2))  
        x = self._depthwise_conv_block(x, self.base_channel * 16, strides=(1, 1))

        x3 = self._branch_block(x, self.base_channel)

        x  = self._depthwise_conv_block(x, self.base_channel * 16, strides=(2, 2)) 
        x4 = self._branch_block(x, self.base_channel)

        extra_layers = [x1, x2, x3,x4]

        confs = []
        locs = []
        head_idx = 0

        assert len(extra_layers) == len(self.num_of_cells)

        for layer in extra_layers:
            conf, loc = self._compute_heads(layer, head_idx, self.num_of_classes, self.num_of_cells)
            confs.append(conf)
            locs.append(loc)
            head_idx += 1

        confs = tf.keras.layers.Concatenate(axis=1, name="face_classes")(confs)
        locs = tf.keras.layers.Concatenate(axis=1, name="face_boxes")(locs)

        predictions = tf.keras.layers.Concatenate(axis=2, name='predictions')([locs, confs])
        model = tf.keras.Model(inputs=inputs, outputs=predictions, name="slim_model")
        return model

