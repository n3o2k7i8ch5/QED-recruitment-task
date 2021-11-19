import gc
import os
from math import log
from statistics import mean

import pandas as pd
import numpy as np
import pickle

import torch
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.autograd import Variable
from torch.nn import LSTM
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from classifier_net import ClassifierNet
from prep_data import prep_data
from prep_lstm_data import prep_lstm_data

LOCALIZED_ALERTS_PATH = 'data/localized_alerts_data.csv'
DUMPS_PATH = 'dumps'
LOGSTACK_PATH = 'dumps/logstack.pkl'

NEURAL_NET_SAVE_PATH = 'neural_net.state'
LSTM_SAVE_PATH = 'lstm.state'

data = pd.read_csv('data/cybersecurity_training.csv', sep='|')

data_x = data
data_x.pop('client_code')
data_y = data_x.pop('notified')

class_ratio = len(data_y[data_y == 1]) / len(data_y)
print(f'y_data "1" to all ratio: {class_ratio}')

data_ids = data_x.pop('alert_ids')
data_ids_num = np.array([i for i in range(len(data_ids.to_numpy().tolist()))], dtype=int)

tmp_data_ids = pd.DataFrame({'id': data_ids, 'id_num': data_ids_num})

if os.path.exists(LOGSTACK_PATH):
    print(f'Logstack exists in {LOGSTACK_PATH}')

    with open(LOGSTACK_PATH, 'rb') as handle:
        logstacks = pickle.load(handle)
else:
    if not os.path.exists(DUMPS_PATH):
        os.mkdir(DUMPS_PATH)
    print(f'Generating logstack from file {LOCALIZED_ALERTS_PATH}')
    localized_alerts = pd.read_csv(LOCALIZED_ALERTS_PATH, sep='|')

    logstacks = prep_lstm_data(localized_alerts, id_dict=tmp_data_ids)

    del localized_alerts

    with open(LOGSTACK_PATH, 'wb') as handle:
        pickle.dump(logstacks, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_x = prep_data(data_x)

print(f'data_x shape = {data_x.shape}')

_limit = int(.9 * len(data_x))

data_ids_train = torch.tensor(data_ids_num[:_limit])
data_ids_test = torch.tensor(data_ids_num[_limit:])

data_x_train = torch.tensor(data_x[:_limit].to_numpy())
data_x_test = torch.tensor(data_x[_limit:].to_numpy())

data_y_train = torch.tensor(data_y[:_limit].to_numpy())
data_y_test = torch.tensor(data_y[_limit:].to_numpy())

# dec tree
'''
dec_tree = DecisionTreeClassifier(class_weight={0: class_ratio, 1: 1 - class_ratio})
dec_tree.fit(data_x_train, data_y_train)

data_pred = dec_tree.predict(data_x_test)
succ = data_pred == data_y_test.numpy()

acc = len(succ[succ == True]) / len(succ)
auc = metrics.roc_auc_score(data_y_test, data_pred)

print('Dec tree acc:', acc)
print('Dec tree auc:', auc)

del dec_tree
del data_pred
del succ
'''

# random forest
'''
rand_forest = RandomForestClassifier(class_weight={0: class_ratio, 1: 1 - class_ratio}, n_estimators=60)
rand_forest.fit(data_x_train, data_y_train)

data_pred = rand_forest.predict(data_x_test)
succ = data_pred == data_y_test.numpy()

acc = len(succ[succ == True]) / len(succ)
auc = metrics.roc_auc_score(data_y_test, data_pred)

print('Random forest acc:', acc)
print('Random forest auc:', auc)

del rand_forest
del data_pred
del succ
'''
# neural network

BATCH_SIZE = 24
LSTM_OUTPUT_SIZE = 160
NUM_LAYERS = 5
LSTM_SEQ_TRUNC = 3200

train_data_loader = DataLoader(TensorDataset(data_ids_train, data_x_train, data_y_train), batch_size=BATCH_SIZE,
                               shuffle=True)
valid_data_loader = DataLoader(TensorDataset(data_ids_test, data_x_test, data_y_test), batch_size=BATCH_SIZE,
                               shuffle=True)

print('cuda available:', torch.cuda.is_available())
print('device in use:', torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device("cuda")

neural_net = ClassifierNet(input_size=len(data_x.columns) + LSTM_OUTPUT_SIZE).to(device)
# neural_net = ClassifierNet(input_size=len(data_x.columns)).to(device)

LSTM_INPUT_SIZE = len(list(logstacks.values())[0].columns)
print(f'LSTM input size: {LSTM_INPUT_SIZE}')

lstm = LSTM(
    input_size=LSTM_INPUT_SIZE,
    hidden_size=LSTM_OUTPUT_SIZE,
    num_layers=NUM_LAYERS,
    dropout=0.3,
    batch_first=True).float().to(device)

del data
del data_ids
del data_x
del data_y

del data_ids_train
del data_ids_test

del data_x_train
del data_x_test

del data_y_train
del data_y_test

del tmp_data_ids

optim_net = AdamW(neural_net.parameters(), lr=.00008)
optim_lstm = AdamW(lstm.parameters(), lr=.00008)

valid_iterator = enumerate(valid_data_loader)


def loss_fun(real: torch.Tensor, pred: torch.Tensor) -> Variable:
    # loss = nn.CrossEntropyLoss(weight=torch.tensor([20., 1.])).to(device)
    # return loss(pred, real)

    real_np = real.detach().cpu().numpy()
    tensor_len = 1 if type(real_np.tolist()) == float else len(real_np.tolist())
    weight = torch.tensor([1.] * tensor_len, device=device)

    if BATCH_SIZE == 1:
        weight *= (1 / class_ratio) if real_np.tolist() == 1 else 1.
    else:
        selector = real_np == 1
        weight[selector] = 1 / class_ratio

    weighted_squares = weight * (real - pred) ** 2
    return weighted_squares.mean()


if True:
    if os.path.exists(NEURAL_NET_SAVE_PATH):
        neural_net.load_state_dict(torch.load(NEURAL_NET_SAVE_PATH))
        print(f'Loaded neural_net from {NEURAL_NET_SAVE_PATH}')

    if os.path.exists(LSTM_SAVE_PATH):
        lstm.load_state_dict(torch.load(LSTM_SAVE_PATH))
        print(f'Loaded lstm from {LSTM_SAVE_PATH}')


def eval_batch(y_data, pred_data):
    pred_data_cpu: np.ndarray = pred_data.detach().cpu().numpy()
    batch_y_cpu: np.ndarray = y_data.detach().cpu().numpy()

    acc_arr = (pred_data_cpu > .5) == (batch_y_cpu > .5)
    if BATCH_SIZE == 1:
        acc = 1 if acc_arr.tolist()[0] else 0
    else:
        acc = len(acc_arr[acc_arr == True]) / len(acc_arr)

    if BATCH_SIZE == 1:
        auc_true = [batch_y_cpu]
        auc_pred = [pred_data_cpu]
    else:
        auc_true = batch_y_cpu.tolist()
        if type(auc_true) == float:
            auc_true = [auc_true]

        auc_pred = pred_data_cpu.tolist()
        if type(auc_pred) == float:
            auc_pred = [auc_pred]

    return acc, auc_true, auc_pred


for epoch in range(30):

    losses = []
    accs = []
    aucs_true = []
    aucs_pred = []

    valid_losses = []
    accs_valid = []
    aucs_true_valid = []
    aucs_pred_valid = []

    for n_batch, (_batch_ids, _batch_x, _batch_y) in enumerate(train_data_loader):

        optim_net.zero_grad()
        optim_lstm.zero_grad()

        vals = []
        for _id in _batch_ids:
            if _id.item() not in logstacks:
                vals.append(torch.tensor(np.array([[0] * LSTM_INPUT_SIZE])))
                continue

            tens = logstacks[_id.item()].to_numpy()
            if len(tens) > LSTM_SEQ_TRUNC:
                tens = tens[-LSTM_SEQ_TRUNC:, :]

            vals.append(torch.tensor(tens))

        vals_padd: torch.Tensor = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True).float().to(device)

        lstm.train(True)
        lstm_all_outs, _ = lstm.forward(vals_padd)
        del vals
        del vals_padd

        lstm_out = lstm_all_outs[:, -1, :]
        del lstm_all_outs

        _batch_x: torch.Tensor = _batch_x.type(torch.FloatTensor).to(device)
        _batch_y: torch.Tensor = _batch_y.type(torch.FloatTensor).to(device)

        # pred_data = neural_net.forward(_batch_x).squeeze()
        neural_net.train(True)
        pred_data = neural_net.forward(torch.cat([_batch_x, lstm_out], dim=1)).squeeze()
        del _batch_x
        del lstm_out

        loss: Variable = loss_fun(_batch_y, pred_data)
        losses.append(loss.item())
        loss.backward()
        optim_net.step()
        optim_lstm.step()

        pred_data[pred_data > 1] = 1
        pred_data[pred_data < 0] = 0

        acc, auc_true, auc_pred = eval_batch(_batch_y, pred_data)
        accs.append(acc)
        aucs_true.extend(auc_true)
        aucs_pred.extend(auc_pred)

        del _batch_y
        del pred_data
        del loss

        try:
            _, (_batch_ids_valid, _batch_x_valid, _batch_y_valid) = next(valid_iterator)
        except StopIteration:
            valid_iterator = enumerate(valid_data_loader)
            _, (_batch_ids_valid, _batch_x_valid, _batch_y_valid) = next(valid_iterator)

        _batch_x_valid = _batch_x_valid.type(torch.FloatTensor).to(device)
        _batch_y_valid = _batch_y_valid.type(torch.FloatTensor).to(device)

        vals_valid = []
        for _id in _batch_ids_valid:
            if _id.item() not in logstacks:
                vals_valid.append(torch.tensor(np.array([[0] * LSTM_INPUT_SIZE])))
                continue

            tens = logstacks[_id.item()].to_numpy()
            if len(tens) > LSTM_SEQ_TRUNC:
                tens = tens[-LSTM_SEQ_TRUNC:, :]

            vals_valid.append(torch.tensor(tens))

        vals_valid_padd: torch.Tensor = torch.nn.utils.rnn.pad_sequence(vals_valid, batch_first=True).float().to(device)
        lstm.train(False)
        lstm_all_outs_valid, _ = lstm.forward(vals_valid_padd)

        del vals_valid
        del vals_valid_padd

        lstm_out_valid = lstm_all_outs_valid[:, -1, :]
        del lstm_all_outs_valid

        # pred_valid_data = neural_net.forward(_batch_x_valid).squeeze()
        neural_net.train(False)
        pred_valid_data = neural_net.forward(torch.cat([_batch_x_valid, lstm_out_valid], dim=1)).squeeze()
        del _batch_x_valid
        del lstm_out_valid

        valid_loss = loss_fun(_batch_y_valid, pred_valid_data)
        valid_losses.append(valid_loss.item())

        pred_valid_data[pred_valid_data > 1] = 1
        pred_valid_data[pred_valid_data < 0] = 0

        acc_v, auc_true_v, auc_pred_v = eval_batch(_batch_y_valid, pred_valid_data)
        accs_valid.append(acc_v)
        aucs_true_valid.extend(auc_true_v)
        aucs_pred_valid.extend(auc_pred_v)

        del _
        del _batch_ids_valid
        del _batch_y_valid

        del pred_valid_data

        del valid_loss

        gc.collect()
        torch.cuda.empty_cache()

        if n_batch % 10 == 0:
            print('.', end=' ')

        if n_batch % 100 == 99:
            auc = metrics.roc_auc_score(aucs_true, aucs_pred)
            aucs_true = []
            aucs_pred = []

            auc_v = metrics.roc_auc_score(aucs_true_valid, aucs_pred_valid)
            aucs_true_valid = []
            aucs_pred_valid = []

            print(
                'epoch:', epoch,
                'n_batch:', n_batch,
                ':: loss (valid):', "{:.3f}".format(round(mean(losses), 3)),
                "{:.3f}".format(round(mean(valid_losses), 3)),

                ':: acc (valid):', "{:.3f}".format(round(mean(accs), 3)), "{:.3f}".format(round(mean(accs_valid), 3)),

                ':: auc (valid):', "{:.3f}".format(round(auc, 3)), "{:.3f}".format(round(auc_v, 3)),
            )
            losses = []
            valid_loss = []
            accs_valid = []

            torch.save(neural_net.state_dict(), NEURAL_NET_SAVE_PATH)
            torch.save(lstm.state_dict(), LSTM_SAVE_PATH)
