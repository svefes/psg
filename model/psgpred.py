import tensorflow as tf
def Conv_1d(inputs: object, model_width: object, kernel: object, strides: object, nl: object) -> object:
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal", kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if nl == 'HS':
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    elif nl == 'RE':
        x = tf.keras.activations.relu(x, max_value=6.0)

    return x

def invert_bottleneck_block(inputs, out_channel, t,  s):
    # inputs: [batch, time, channel]
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = Conv_1d(inputs=inputs, model_width=tchannel, kernel=1, strides=1, nl='RE')
    x = tf.keras.layers.SeparableConv1D(filters=tchannel, kernel_size=3, strides=s, depth_multiplier=1, padding='same',kernel_regularizer='l2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(out_channel, 1, strides=1, padding='same', kernel_regularizer='l2')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if s == 1:
        x = tf.keras.layers.Add()([x, inputs])

    return x

class PsgPred:
    def __init__(self, input_shape, input_channels, drop_out):
        #channels = ['EOG LOC-M2', 'EOG ROC-M1', 'EMG Chin1-Chin2', 'EEG F3-M2', 'EEG F4-M1', 'EEG C3-M2',
        #            'EEG C4-M1', 'EEG O1-M2', 'EEG O2-M1', 'EEG CZ-O1', 'ECG EKG2-EKG', 'Resp Airflow',
        #            'Resp Thoracic', 'Resp Abdominal', 'SpO2', 'Pressure']
        self.input_channels = tf.constant(input_channels)
        self.input_shape = input_shape
        self.dropout_rate = drop_out

    def model_setup(self):
        # [batch, time, channels]
        inputs = tf.keras.Input(self.input_shape)
        # remove unwanted channels
        inputs_down = tf.gather(inputs, indices=self.input_channels, axis=-1)
        # Channel-Mixing
        x = Conv_1d(inputs_down, model_width=32, kernel=1, strides=1, nl='RE')
        # CNN for feature extraction
        x = invert_bottleneck_block(x, out_channel=16, t=1, s=2)
        x = invert_bottleneck_block(x, out_channel=32, t=6, s=2)
        x = invert_bottleneck_block(x, out_channel=32, t=6, s=1)
        x = invert_bottleneck_block(x, out_channel=64, t=6, s=2)
        x = invert_bottleneck_block(x, out_channel=64, t=6, s=1)
        x = invert_bottleneck_block(x, out_channel=128, t=6, s=2)
        # x: [batch, time/16, 128]
        x = tf.keras.layers.AveragePooling1D(pool_size=20, strides=20)(x)
        # Bi-LSTM
        forward_layer = tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer='l2', name="fwd_lstm")
        backward_layer = tf.keras.layers.LSTM(128, return_sequences=True, go_backwards=True, kernel_regularizer='l2', name="bwd_lstm")
        x = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(x)
        # x: [batch, time/16, 2*128]
        # Additive Attention (input: 256, hidden: 512)
        u = tf.keras.layers.Dense(512, activation='tanh', kernel_regularizer='l2')(x)
        a = tf.nn.softmax(tf.keras.layers.Dense(units=1, activation='linear', use_bias=False, kernel_regularizer='l2')(u), axis=1)
        s = tf.math.reduce_sum(a*x, axis=1)
        # a:[batch, time/16, 1], s:[batch, 2*128]
        # Dense squeezing
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2')(s)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.squeeze(tf.keras.layers.Dense(1, activation='linear', kernel_regularizer='l2')(x))
        model = tf.keras.Model(inputs, outputs)
        return model