# Pediatric age predictor based on PSG snippets

train_model.py contains the relevant code to train, validate and test (cross-validation) the (pediatric) age predictor.
It depends on TensorFlow 2 (with GPU support).

The data needs to be stored as .tfrecord files in exactly 10 directories of similar size that lie in  all\_data\_dir.
The individual samples in such a record need to contain the following fields:
 	context_dict = {'subject id': tf.io.VarLenFeature(dtype=tf.string),
                        'signals': tf.io.VarLenFeature(dtype=tf.string),
                        'age': tf.io.VarLenFeature(dtype=tf.int64),
                        'start time': tf.io.VarLenFeature(dtype=tf.string),
                        'sleep stages': tf.io.VarLenFeature(dtype=tf.int64)}          
        signal_dict = {'signals': tf.io.FixedLenSequenceFeature([num_channels],dtype=tf.float32)}
        
Due to data access restrictions, the code for data pre-processing and post-processing cannot be published.
After getting credentialed access to https://physionet.org/content/nch-sleep/3.1.0/ (managed by PhysioNet), the code for data processing can be made available upon request.
