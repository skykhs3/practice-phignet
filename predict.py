import argparse
import os, json
import numpy as np
import tensorflow as tf

from src.utils import *
from src.layers import *

os.environ['CUDA_VISIBLE_DEVICES'] = ''
tf.config.list_physical_devices('CPU')
L_pad_go = 1024
L_pad_ec = 1002

dirt_param = './models/'

mf_model = tf.keras.models.load_model(dirt_param+'PhiGnet_GCN_0.2_mf.hdf5',
                                       custom_objects={'GraphConv1': GraphConv1,
                                                       'GraphConv2': GraphConv2,
                                                       'Reshape_out': Reshape_out,
                                                       'get_sum': get_sum
                                                       }
                                     )

bp_model = tf.keras.models.load_model(dirt_param+'PhiGnet_GCN_0.2_bp.hdf5',
                                       custom_objects={'GraphConv1': GraphConv1,
                                                       'GraphConv2': GraphConv2,
                                                       'Reshape_out': Reshape_out,
                                                       'get_sum': get_sum
                                                       }
                                     )

cc_model = tf.keras.models.load_model(dirt_param+'PhiGnet_GCN_0.2_cc.hdf5',
                                       custom_objects={'GraphConv1': GraphConv1,
                                                       'GraphConv2': GraphConv2,
                                                       'Reshape_out': Reshape_out,
                                                       'get_sum': get_sum
                                                       }
                                     )

ec_model = tf.keras.models.load_model(dirt_param+'PhiGnet_GCN_0.2_ec.hdf5',
                                       custom_objects={'GraphConv1': GraphConv1,
                                                       'GraphConv2': GraphConv2,
                                                       'Reshap_out': Reshape_out,
                                                       'get_sum': get_sum
                                                       }
                                     )


with open(dirt_param+'PhiGnet_GCN_0.2_mf_model_params.json') as mf_file:
    mfdata = json.load(mf_file)
mf_gonames = np.asarray(mfdata['gonames'])
mf_goterms = np.asarray(mfdata['goterms'])
mf_thresh  = 0.4*np.ones(len(mf_goterms))

with open(dirt_param+'PhiGnet_GCN_0.2_bp_model_params.json') as bp_file:
    bpdata = json.load(bp_file)
bp_gonames = np.asarray(bpdata['gonames'])
bp_goterms = np.asarray(bpdata['goterms'])
bp_thresh  = 0.4*np.ones(len(bp_goterms))

with open(dirt_param+'PhiGnet_GCN_0.2_cc_model_params.json') as cc_file:
    ccdata = json.load(cc_file)
cc_gonames = np.asarray(ccdata['gonames'])
cc_goterms = np.asarray(ccdata['goterms'])
cc_thresh  = 0.4*np.ones(len(cc_goterms))

with open(dirt_param+'PhiGnet_GCN_0.2_ec_model_params.json') as ec_file:
    ecdata = json.load(ec_file)
ec_names = np.asarray(ecdata['gonames'])
ec_numbers = np.asarray(ecdata['goterms'])
ec_thresh  = 0.4*np.ones(len(ec_numbers))

def feature_padding(evc, rc, esm1b, sequence, cut_thresh, ont):

    if ont == 'ec':
        Lmax = L_pad_ec
    elif ont != 'ec':
        Lmax = L_pad_go
    
    A = np.double(evc > cut_thresh)
    m = evc.shape[0]
    n = Lmax - m
    A = tf.pad(A, [[0, n], [0, n]])
    A = np.array(A)

    rc = np.double(rc > cut_thresh)
    rc = tf.pad(rc, [[0, n], [0, n]])
    rc = np.array(rc)
    
    esm = tf.pad(esm1b, [[0, n-2], [0, 0]])
    esm = np.array(esm)
    seq = sequence

    S = seq2onehot(seq)
    a_s = S.shape[0]
    l_s = Lmax-a_s
    S = tf.pad(S, [[0, l_s], [0, 0]])
    S = np.array(S)
    S = S.reshape(1, *S.shape)
    A = A.reshape(1, *A.shape)
    rc = rc.reshape(1, *rc.shape)
    esm = esm.reshape(1, *esm.shape)

    return A, S, rc, esm, seq

def get_inputs(file, cut_thresh, ont):

    data = np.load(file)
    evc = data['coupling']
    rc = data['rc']
    seq = str(data['seq'])
    esm1b = data['esm_1b']
    EVCs, S, RCs, ESM1b, seq_pad = feature_padding(evc, rc, esm1b, seq, cut_thresh, ont)

    return EVCs, S, RCs, ESM1b, seq_pad, seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('-ont', '--ontology', type=str, default='mf', choices=['mf', 'bp', 'cc', 'ec'], help='the classes of GO terms')
    parser.add_argument('-p', '--path_inputs', type=str, default='./examples/', help='the path save the inputs of PhiGnet')
    parser.add_argument('-j', '--job_id', type=str, default='', help='the name of protein whose function is to be predicted')
    parser.add_argument('-ct', '--cut_thresh',  type=float, default=0.2, help='the threshold of filtering EVCs and RCs')
    parser.add_argument('-d', '--dirt',   type=str, default='', help='the path of output')
    args  = parser.parse_args()
    
    dirt  = args.dirt + ('/' if args.dirt.strip()[-1]  != '/' else '')
    jobid = args.job_id
    ont  = args.ontology
    cut_thresh = args.cut_thresh

    path_inputs = args.path_inputs + ('/' if args.dirt.strip()[-1]  != '/' else '') + jobid + '.npz'
    EVCs, S, RCs, ESM1b, seq_pad, sequence = get_inputs(path_inputs, cut_thresh, ont)

    csv_write([], ont.upper(), jobid, dirt, mod = 'w+') # Initialize

    prot2goterms = {}
    prot2goterms[jobid] = []
    if ont == 'mf':
        threshold = mf_thresh
        y = mf_model([EVCs, RCs, ESM1b, S], training=False).numpy()[:, :, 0].reshape(-1)
    if ont == 'bp':
        threshold = bp_thresh
        y = bp_model([EVCs, RCs, ESM1b, S], training=False).numpy()[:, :, 0].reshape(-1)
    if ont == 'cc':
        threshold = cc_thresh
        y = cc_model([EVCs, RCs, ESM1b, S], training=False).numpy()[:, :, 0].reshape(-1)
    if ont == 'ec':
        threshold = ec_thresh
        y = ec_model([EVCs, RCs, ESM1b, S], training=False).numpy()[:, :, 0].reshape(-1)

    ont_idx = np.where((y >= threshold) == True)[0]

    for idx in ont_idx:
        if ont == 'mf':
            ont_terms = mf_goterms[idx]
            ont_names = mf_gonames[idx]
            t_model = mf_model
        if ont == 'bp':
            ont_terms = bp_goterms[idx]
            ont_names = bp_gonames[idx]
            t_model = bp_model
        if ont == 'cc':
            ont_terms = cc_goterms[idx]
            ont_names = cc_gonames[idx]
            t_model = cc_model
        if ont == 'ec':
            ont_terms = ec_numbers[idx]
            ont_names = ec_names[idx]
            t_model = ec_model

        prot2goterms[jobid].append((ont_terms, ont_names, float(y[idx])))

    csv_write(prot2goterms, ont.upper(), jobid, dirt, mod = 'a')
    
