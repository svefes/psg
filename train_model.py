#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import tensorflow as tf
import glob
from model import psgpred
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"]="1"
all_data_dir = "./data/Sleep_Data/"
channels = 16
batch = 50
epochs = 100
standardise = True



def _gen_parse_function(num_channels, test=False):
    def _parse_function(example):
        context_dict = {'subject id': tf.io.VarLenFeature(dtype=tf.string),
                        'signals': tf.io.VarLenFeature(dtype=tf.string),
                        'age': tf.io.VarLenFeature(dtype=tf.int64),
                        'start time': tf.io.VarLenFeature(dtype=tf.string),
                        'sleep stages': tf.io.VarLenFeature(dtype=tf.int64)}
        # extend the dict for more data feature lists
        signal_dict = {'signals': tf.io.FixedLenSequenceFeature([num_channels],dtype=tf.float32)}
        context, signal = tf.io.parse_single_sequence_example(example, context_features=context_dict, sequence_features=signal_dict)
        return signal['signals'], tf.cast(tf.sparse.to_dense(context["age"])/365, tf.float32)
    if test:
        def _parse_function(example):
            context_dict = {'subject id': tf.io.VarLenFeature(dtype=tf.string),
                            'signals': tf.io.VarLenFeature(dtype=tf.string),
                            'age': tf.io.VarLenFeature(dtype=tf.int64),
                            'start time': tf.io.VarLenFeature(dtype=tf.string),
                            'sleep stages': tf.io.VarLenFeature(dtype=tf.int64)}
            # extend the dict for more data feature lists
            signal_dict = {'signals': tf.io.FixedLenSequenceFeature([num_channels], dtype=tf.float32)}
            context, signal = tf.io.parse_single_sequence_example(example, context_features=context_dict,
                                                                  sequence_features=signal_dict)
            return signal['signals'], tf.cast(tf.sparse.to_dense(context["age"])/365, tf.float32), tf.sparse.to_dense(context["subject id"])
    return _parse_function

def produce_age_sel(max_age):
    ma = tf.constant(max_age, dtype=tf.float32)
    def age_sel(signal, age, *args):
        return tf.squeeze(tf.math.less_equal(age, ma))
    return age_sel


def create_data(max_age):

    cv_dirs = next(os.walk(all_data_dir, topdown=True))[1]
    assert len(cv_dirs) == 10 and cv_id < 5
    test_bins = cv_dirs[cv_id * 2:cv_id * 2 + 2]
    valid_bin = [cv_dirs[(cv_id * 2 + 2) % 10]]
    start_back_train = cv_id * 2 + 3
    if start_back_train > 9:
        end = []
        start = cv_dirs[1:8]
    else:
        end = cv_dirs[start_back_train:]
        start = cv_dirs[:cv_id * 2]
    train_bins = start + end

    inputs_train = []
    for x in train_bins:
        inputs_train += glob.glob(all_data_dir + "/" + x + "/*.tfrecord")
    inputs_valid = []
    for x in valid_bin:
        inputs_valid += glob.glob(all_data_dir + "/" + x + "/*.tfrecord")
    inputs_test = []
    for x in test_bins:
        inputs_test += glob.glob(all_data_dir + "/" + x + "/*.tfrecord")

    age_sel = produce_age_sel(max_age=max_age)

    parse_func = _gen_parse_function(channels)
    test_parse_func = _gen_parse_function(channels, test=True)
    raw_trdata = tf.data.TFRecordDataset(inputs_train)
    train_set = raw_trdata.map(parse_func).filter(age_sel).shuffle(10000).batch(batch, drop_remainder=True).prefetch(buffer_size=10)
    raw_vdata = tf.data.TFRecordDataset(inputs_valid)
    valid_set = raw_vdata.map(parse_func).filter(age_sel).batch(10, drop_remainder=True).prefetch(buffer_size=10)
    raw_tsdata = tf.data.TFRecordDataset(inputs_test)
    test_set = raw_tsdata.map(test_parse_func).filter(age_sel).batch(10, drop_remainder=True).prefetch(buffer_size=10)

    resmeans = None
    resstds = None
    if standardise:
        means = [tf.keras.metrics.Mean() for i in range(channels)]
        vars = [tf.keras.metrics.MeanSquaredError() for i in range(channels)]
        for x, age in train_set:
            for i, channel in enumerate(tf.split(x, num_or_size_splits=channels, axis=-1)):
                means[i].update_state(channel)
        for x, age in train_set:
            for i, channel in enumerate(tf.split(x, num_or_size_splits=channels, axis=-1)):
                # biased estimator :(
                vars[i].update_state(channel, means[i].result())
        resmeans = [m.result() for m in means]
        resvars = [s.result() for s in vars]
        resstds = [tf.sqrt(s) for s in resvars]
        st_layer = tf.keras.layers.Normalization(mean=resmeans, variance=resvars)
        train_set = train_set.map(lambda x, y: (st_layer(x), y))
        test_set = test_set.map(lambda x, y, z: (st_layer(x), y, z))
        valid_set = valid_set.map(lambda x, y: (st_layer(x), y))
    return train_set, test_set, valid_set, resmeans, resstds


def make_or_restore_model(channels_of_interest=None):
    checkpoint_dir = log_dir+"/ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints and restore:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return tf. keras.models.load_model(latest_checkpoint), checkpoint_dir
    print("Creating a new model")
    if channels_of_interest is None:
        channels_of_interest = range(16)
    model = psgpred.PsgPred(input_shape=[5 * 60 * 128, 16], input_channels=channels_of_interest, drop_out=0.3).model_setup()
    opti = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opti,
        # Huber loss limit was 5 before that version
        loss=tf.keras.losses.Huber(delta=5), metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model, checkpoint_dir

def train_model(channels_of_interest=None):

    model, ckpt_dir = make_or_restore_model(channels_of_interest)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_dir + "/ckpt-loss={loss:.2f}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir+"/tb",
            histogram_freq=0,  # How often to log histogram visualizations
            embeddings_freq=0,  # How often to log embedding visualizations
            update_freq="epoch",
        )
    ]

    model.fit(
        x=train_set,
        epochs=epochs,
        validation_data=valid_set,
        callbacks=callbacks)

    return model

def test_model(model):
    with open(log_dir + "/res.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ID", "Age", "Predicted Age", "Age-Predicted"])
        oi = ""
        coi = 1
        real_age = 0
        ensemble_age = 0
        for signal, age, id in test_set:
            pred_age = model(signal)
            ids = id.numpy()
            ages = age.numpy()
            pred_ages = pred_age.numpy()
            for i,a,p in zip(ids, ages, pred_ages):
                if oi == str(i[0]):
                    ensemble_age += p
                    coi += 1
                else:
                    writer.writerow([oi,real_age,ensemble_age/coi,real_age-ensemble_age/coi])
                    oi = str(i[0])
                    coi = 0
                    real_age = a[0]
                    ensemble_age = 0
        writer.writerow([oi, real_age, ensemble_age / coi, real_age - ensemble_age / coi])


# train
for dt_sh, cv_id in [("large_noreg_huber2", 0), ("large_noreg_huber2", 1), ("large_noreg_huber2", 2), ("large_noreg_huber2", 3), ("large_noreg_huber2", 4)]:
    log_dir = "logs/{}_{}".format(dt_sh, cv_id)
    restore = False
    train_set, test_set, valid_set, means, stds = create_data()
    trained_model = train_model()
    test_model(trained_model)
