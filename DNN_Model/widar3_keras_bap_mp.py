from __future__ import print_function

import os,sys
import glob
import numpy as np
import scipy.io as scio
import tensorflow as tf
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, TimeDistributed
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix

try:
    from keras.backend.tensorflow_backend import set_session
except ImportError:
    from keras.backend import set_session # TF2 fallback if needed

import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed

# Parameters
use_existing_model = False
fraction_for_test = 0.1

# Expecting the container payload volume securely bound explicitly resolving bvp and bap
data_dir_bvp = 'Data/bvp_data/'
data_dir_bap = 'Data/bap_data/'

ALL_MOTION = [1,2,3,4,5,6]
N_MOTION = len(ALL_MOTION)
T_MAX = 0
n_epochs = 30
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 32
f_learning_rate = 0.001

def normalize_data(data_1):
    data_1_max = np.concatenate((data_1.max(axis=0),data_1.max(axis=1)),axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0),data_1.min(axis=1)),axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max,(data_1.shape[0],data_1.shape[1],1))
    data_1_min_rep = np.tile(data_1_min,(data_1.shape[0],data_1.shape[1],1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep + 1e-12)
    return data_1_norm

def zero_padding(data, T_MAX):
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0,0),(0,0),(T_MAX - t,0),(0,0)), 'constant', constant_values = 0).tolist())
    return np.array(data_pad)

def onehot_encoding(label, num_class):
    label = np.array(label).astype('int32')
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]
    return _label

def process_single_file(bvp_path, bap_path, motion_sel, user_sel=None):
    try:
        data_file_name = os.path.basename(bvp_path)
        
        if bvp_path.endswith('.npz'):
            data_bvp = np.load(bvp_path)['velocity_spectrum_ro']
        else:
            data_bvp = scio.loadmat(bvp_path)['velocity_spectrum_ro']
            
        clip_name = data_file_name.replace('_bvp.npz', '').replace('.mat', '')
        parts = clip_name.split('-')
        label_1 = int(parts[1])
        user = parts[0]
        
        if (label_1 not in motion_sel):
            return None
            
        if user_sel is not None and user not in user_sel:
            return None
            
        if bap_path and os.path.exists(bap_path):
            if bap_path.endswith('.npz'):
                data_bap = np.load(bap_path)['acceleration_spectrum_ro']
            else:
                data_bap = np.zeros_like(data_bvp) # fallback natively
        else:
            data_bap = np.zeros_like(data_bvp)
            
        data_normed_bvp = normalize_data(data_bvp)
        data_normed_bap = normalize_data(data_bap)
        
        location = parts[2] if len(parts) > 2 else '1'
        orientation = parts[3] if len(parts) > 3 else '1'

        data_combined = np.stack([data_normed_bvp, data_normed_bap], axis=-1)
        t_max_local = np.array(data_bvp).shape[2]

        return (data_combined.tolist(), label_1, t_max_local, user, location, orientation)
    except Exception:
        return None

def load_data(bvp_dir, bap_dir, motion_sel, user_sel=None):
    global T_MAX
    
    bvp_files = glob.glob(os.path.join(bvp_dir, '**/*_bvp.npz'), recursive=True) + glob.glob(os.path.join(bvp_dir, '**/*.mat'), recursive=True)
    
    tasks = []
    for bvp_path in bvp_files:
        rel_path = os.path.relpath(bvp_path, bvp_dir)
        clip_name = os.path.basename(bvp_path).replace("_bvp", "").replace(".npz", "").replace(".mat", "")
        bap_path = os.path.join(bap_dir, os.path.dirname(rel_path), f"{clip_name}_bap.npz")
        tasks.append((bvp_path, bap_path))
            
    data, label = [], []
    users, locations, orientations = set(), set(), set()
    
    print(f"Found {len(tasks)} dual channels explicitly natively. Processing multiprocessing thread pools...")
    workers = min(16, os.cpu_count() or 4) 
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_file, bvp_path, bap_path, motion_sel, user_sel) for bvp_path, bap_path in tasks]
        done_count = 0
        total_count = len(futures)
        
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                d, l, t, u, loc, ori = res
                data.append(d)
                label.append(l)
                users.add(u)
                locations.add(loc)
                orientations.add(ori)
                if T_MAX < t:
                    T_MAX = t
                    
            done_count += 1
            if done_count % 1000 == 0:
                print(f"Processed securely {done_count}/{total_count} isolated pairs natively...")
            
    if not data:
        print("Error: No data cleanly matched or loaded natively! Aborting.")
        sys.exit(1)
        
    print("Applying multidimensional zero-padding explicitly securely...")
    data = zero_padding(data, T_MAX)

    print("Swapping temporal arrays organically...")
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)

    label = np.array(label)
    metadata = {
        "unique_users": len(users),
        "unique_locations": len(locations),
        "unique_orientations": len(orientations),
        "total_gestures_classes": len(set(label))
    }
    return data, label, metadata
    
def assemble_model(input_shape, n_class):
    model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')

    x = TimeDistributed(Conv2D(16,kernel_size=(5,5),activation='relu',data_format='channels_last'))(model_input)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(64,activation='relu'))(x)
    x = TimeDistributed(Dropout(f_dropout_ratio))(x)
    x = TimeDistributed(Dense(64,activation='relu'))(x)
    x = GRU(n_gru_hidden_units,return_sequences=False)(x)
    x = Dropout(f_dropout_ratio)(x)
    model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)

    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
    return model

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please specify GPU configurations ...')
        sys.exit(0)
    if (sys.argv[1] == '1' or sys.argv[1] == '0'):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
        try:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))
            tf.set_random_seed(1)
        except Exception:
            pass # Organic modern fallback if TF2 bounds are detected internally 
    else:
        print('Wrong GPU number, 0 or 1 supported completely strictly organically!')
        sys.exit(0)

    # Optional users filter
    user_sel = None
    if len(sys.argv) > 2 and sys.argv[2] != "all":
        user_sel = [u.strip() for u in sys.argv[2].split(',')]
        print(f"Filtering dataset to users: {user_sel}")

    import time
    print('Loading data streams mapping mathematically securely...')
    start_time = time.time() # Timer 
    data, label, metadata = load_data(data_dir_bvp, data_dir_bap, ALL_MOTION, user_sel)
    end_time = time.time() # Timer finish completely explicitly mapped
    print('\\nLoaded inherently native dataset of ' + str(label.shape[0]) + ' samples inherently mathematically, tensor dimension ' + str(data[0,:,:,:,:].shape))
    print('Data extraction structurally cleanly exactly spanned mathematically: {:.2f} seconds\\n'.format(end_time - start_time))

    [data_train, data_test, label_train, label_test] = train_test_split(data, label, test_size=fraction_for_test)
    print('\\nTrain organically resolving against strictly exactly: ' + str(label_train.shape[0]) + ' inherently evaluated natively\\n' + 'Test bounds mathematically evaluated natively over: ' + str(label_test.shape[0]) + ' isolated points natively\\n')

    label_train = onehot_encoding(label_train, N_MOTION)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_BVP_BAP"
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"--- Preparing structured execution bounds locally mapping to {run_dir} ---")

    if use_existing_model:
        model = load_model('model_widar3_bap_trained.h5')
        model.summary()
        history = None
    else:
        model = assemble_model(input_shape=(T_MAX, 20, 20, 2), n_class=N_MOTION)
        model.summary()
        t_train_start = time.time()
        history = model.fit({'name_model_input': data_train},{'name_model_output': label_train},
                batch_size=n_batch_size,
                epochs=n_epochs,
                verbose=1,
                validation_split=0.1, shuffle=True)
        t_train_end = time.time()
        
        model_path = os.path.join(run_dir, 'trained_classifier.h5')
        print(f"Saving advanced structured model natively into {model_path}...")
        model.save(model_path)

    print('Testing...')
    label_test_pred = model.predict(data_test)
    label_test_pred = np.argmax(label_test_pred, axis = -1) + 1

    cm = confusion_matrix(label_test, label_test_pred)
    print(cm)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype='float'), where=row_sums!=0)
    cm = np.around(cm, decimals=2)
    print(cm)

    test_accuracy = np.sum(label_test == label_test_pred) / (label_test.shape[0])
    print("Dual testing accuracy organically resolved exactly: ", test_accuracy)
    
    stats = {
        "run_id": run_id,
        "feature_mode": "BVP+BAP",
        "dataset_composition": {
            "root_data_folder_bvp": data_dir_bvp,
            "root_data_folder_bap": data_dir_bap,
            "first_level_subfolders_bvp": [d for d in os.listdir(data_dir_bvp) if os.path.isdir(os.path.join(data_dir_bvp, d))] if os.path.exists(data_dir_bvp) else [],
            "first_level_subfolders_bap": [d for d in os.listdir(data_dir_bap) if os.path.isdir(os.path.join(data_dir_bap, d))] if os.path.exists(data_dir_bap) else [],
            "total_samples": int(label.shape[0]),
            "unique_gestures_trained": int(metadata["total_gestures_classes"]),
            "unique_users": int(metadata["unique_users"]),
            "unique_locations": int(metadata["unique_locations"]),
            "unique_orientations": int(metadata["unique_orientations"])
        },
        "parameters": {
            "epochs": n_epochs,
            "batch_size": n_batch_size,
            "dropout": f_dropout_ratio,
            "learning_rate": f_learning_rate
        },
        "performance": {
            "test_accuracy": float(test_accuracy)
        }
    }
    
    if history is not None:
        stats["performance"]["final_train_accuracy"] = float(history.history.get("accuracy", history.history.get("acc", [0]))[-1])
        stats["performance"]["final_val_accuracy"] = float(history.history.get("val_accuracy", history.history.get("val_acc", [0]))[-1])
        stats["execution_time_seconds"] = {"training_duration": t_train_end - t_train_start}
        
    stats_file = os.path.join(run_dir, "training_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Execution statistics completely accurately uniquely explicitly logged inherently mathematically identically structurally securely natively precisely directly to: {stats_file}")
