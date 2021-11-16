import gc
import os
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
from data_prep import prep_data
from ip_to_tensor import ip_to_bin_list

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

    logstacks = prep_data(localized_alerts,id_dict=tmp_data_ids)

    del localized_alerts

    with open(LOGSTACK_PATH, 'wb') as handle:
        pickle.dump(logstacks, handle, protocol=pickle.HIGHEST_PROTOCOL)

dstip_lists = [ip_to_bin_list(ip) for ip in data_x['ip']]
dstip_df = pd.DataFrame(dstip_lists, columns=[f'p_{i}' for i in range(40)])
dstip_df.index = data_x.index
data_x = pd.concat([data_x, dstip_df], axis=1)
data_x.pop('ip')

data_x = pd.get_dummies(data_x, columns=[
    'categoryname', 'ipcategory_name', 'ipcategory_scope',
    'parent_category', 'grandparent_category', 'overallseverity', 'weekday',
    'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10',
    'score',

    'alerttype_cd', 'direction_cd', 'eventname_cd', 'severity_cd',
    'reportingdevice_cd', 'devicetype_cd', 'devicevendor_cd', 'domain_cd',
    'protocol_cd', 'username_cd', 'srcipcategory_cd', 'dstipcategory_cd',
    'isiptrusted', 'untrustscore', 'flowscore', 'trustscore', 'enforcementscore',

    'dstipcategory_dominate', 'srcipcategory_dominate',
    'dstportcategory_dominate', 'srcportcategory_dominate',
    'p6', 'p9', 'p5m', 'p5w', 'p5d', 'p8m', 'p8w', 'p8d'
], prefix_sep='$', dummy_na=True)

data_x.drop(columns=['start_hour', 'start_minute', 'start_second'])

print(f'data_x shape = {data_x.shape}')

_limit = int(.8 * len(data_x))

data_ids_train = torch.tensor(data_ids_num[:_limit])
data_ids_test = torch.tensor(data_ids_num[_limit:])

data_x_train = torch.tensor(data_x[:_limit].to_numpy())
data_x_test = torch.tensor(data_x[_limit:].to_numpy())

data_y_train = torch.tensor(data_y[:_limit].to_numpy())
data_y_test = torch.tensor(data_y[_limit:].to_numpy())

# dec tree
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

# random forest
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

# neural network

BATCH_SIZE = 2
LSTM_OUTPUT_SIZE = 20
NUM_LAYERS = 4

train_data_loader = DataLoader(TensorDataset(data_ids_train, data_x_train, data_y_train), batch_size=BATCH_SIZE,
                               shuffle=True)
valid_data_loader = DataLoader(TensorDataset(data_ids_test, data_x_test, data_y_test), batch_size=BATCH_SIZE,
                               shuffle=True)

print('cuda available:', torch.cuda.is_available())
print('device in use:', torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device("cuda")

neural_net = ClassifierNet(input_size=len(data_x.columns) + LSTM_OUTPUT_SIZE).to(device)

LSTM_INPUT_SIZE = len(list(logstacks.values())[0].columns)
print(f'LSTM input size: {LSTM_INPUT_SIZE}')

lstm = LSTM(
    input_size=LSTM_INPUT_SIZE,
    hidden_size=LSTM_OUTPUT_SIZE,
    num_layers=NUM_LAYERS,
    # dropout=0.2,
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

optim_net = AdamW(neural_net.parameters(), lr=.0001)
optim_lstm = AdamW(lstm.parameters(), lr=.0001)

valid_iterator = enumerate(valid_data_loader)


def loss_fun(real: torch.Tensor, pred: torch.Tensor) -> Variable:
    # loss = nn.CrossEntropyLoss(weight=torch.tensor([20., 1.])).to(device)
    # return loss(pred, real)

    real_np = real.detach().cpu().numpy()
    weight = torch.tensor([1] * (1 if type(real_np) == float else len(real_np)), device=device)
    selector = real_np == 1
    weight[selector] = 20
    return (weight * (real - pred) ** 2).mean()

if False:
    if os.path.exists(NEURAL_NET_SAVE_PATH):
        neural_net.load_state_dict(torch.load(NEURAL_NET_SAVE_PATH))
        print(f'Loaded neural_net from {NEURAL_NET_SAVE_PATH}')

    if os.path.exists(LSTM_SAVE_PATH):
        lstm.load_state_dict(torch.load(LSTM_SAVE_PATH))
        print(f'Loaded lstm from {LSTM_SAVE_PATH}')

for epoch in range(15):

    losses = []
    valid_losses = []
    accs = []
    aucs_true = []
    aucs_pred = []

    for n_batch, (_batch_ids, _batch_x, _batch_y) in enumerate(train_data_loader):

        optim_net.zero_grad()
        optim_lstm.zero_grad()

        vals = [torch.tensor(logstacks[_id.item()].to_numpy() if _id.item() in logstacks else np.array([[0] * LSTM_INPUT_SIZE])) for _id in _batch_ids]
        vals_padd: torch.Tensor = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True).float().to(device)
        #print(vals_padd.size())

        lstm_all_outs, _ = lstm.forward(vals_padd)
        del vals
        del vals_padd

        lstm_out = lstm_all_outs[:, -1, :]
        del lstm_all_outs

        _batch_x = _batch_x.type(torch.FloatTensor).to(device)
        _batch_y = _batch_y.type(torch.FloatTensor).to(device).squeeze()

        pred_data = neural_net.forward(torch.cat([_batch_x, lstm_out], dim=1)).squeeze()
        del _batch_x
        del lstm_out

        loss: Variable = loss_fun(_batch_y, pred_data)
        losses.append(loss.item())
        loss.backward()
        optim_net.step()
        optim_lstm.step()

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

        vals = [torch.tensor(logstacks[_id.item()].to_numpy()) for _id in _batch_ids_valid]
        vals_padd: torch.Tensor = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True).float().to(device)
        lstm_all_outs_valid, _ = lstm.forward(vals_padd)

        del vals
        del vals_padd

        lstm_out_valid = lstm_all_outs_valid[:, -1, :]
        del lstm_all_outs_valid

        pred_valid_data = neural_net.forward(torch.cat([_batch_x_valid, lstm_out_valid], dim=1)).squeeze()
        del _batch_x_valid
        del lstm_out_valid

        valid_loss = loss_fun(_batch_y_valid, pred_valid_data)
        valid_losses.append(valid_loss.item())

        pred_valid_data_cpu: np.ndarray = pred_valid_data.detach().cpu().numpy()
        _batch_y_valid_cpu: np.ndarray = _batch_y_valid.detach().cpu().numpy()

        acc = (pred_valid_data_cpu > .5) == (_batch_y_valid_cpu > .5)
        accs.append(len(acc[acc == True]) / len(acc))

        if BATCH_SIZE == 1:
            aucs_true.append(_batch_y_valid_cpu)
            aucs_pred.append(pred_valid_data_cpu)
        else:
            aucs_true.extend(_batch_y_valid_cpu.tolist())
            aucs_pred.extend(pred_valid_data_cpu.tolist())

        del _
        del _batch_ids_valid
        del _batch_y_valid

        del pred_valid_data
        del pred_valid_data_cpu
        del _batch_y_valid_cpu

        del valid_loss

        gc.collect()
        torch.cuda.empty_cache()

        if n_batch % 500 == 499:
            auc = metrics.roc_auc_score(aucs_true, aucs_pred)
            auc_bin = metrics.roc_auc_score(aucs_true, [(0 if x < .5 else 1) for x in aucs_pred])
            aucs_true = []
            aucs_pred = []

            print(
                'epoch:', epoch,
                'n_batch:', n_batch,
                'loss:', "{:.6f}".format(round(mean(losses), 6)),
                'valid_loss:', "{:.6f}".format(round(mean(valid_losses), 6)),
                'acc:', mean(accs),
                'auc:', auc,
                'auc_bin:', auc_bin
            )
            losses = []
            valid_loss = []
            accs = []

            torch.save(neural_net.state_dict(), NEURAL_NET_SAVE_PATH)
            torch.save(lstm.state_dict(), LSTM_SAVE_PATH)
