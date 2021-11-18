from math import log, copysign

import pandas as pd

from ip_to_tensor import ip_to_bin_list, port_to_bin_list


def __protocol_simpl(x):
    if type(x) != str:
        return 'other'

    if x == '-':
        return 'other'

    try:
        int(x)
        return 'other'
    except Exception:
        pass

    x = x.lower()
    x = x.replace('\\', '')
    y = x.replace('-', '%') \
        .replace('_', '%') \
        .replace('/', '%') \
        .replace('(', '%') \
        .replace(')', '%')

    if '%udp' in y or 'udp%' in y or y == 'udp':
        return 'udp'
    elif '%tcp' in y or 'tcp%' in y or y == 'tcp':
        return 'tcp'
    elif '%http' in y or 'http%' in y or y == 'http':
        return 'http'
    elif '%https' in y or 'https%' in y or y == 'https':
        return 'https'
    elif '%dns' in y or 'dns%' in y or y == 'dns':
        return 'dns'
    else:
        return 'other'


def prep_lstm_data(data: pd.DataFrame, id_dict):
    ids = data['alert_ids'].tolist()

    data['protocol'] = data['protocol'].apply(__protocol_simpl)
    data['alerttime'] = data['alerttime'].apply(lambda x: copysign(1, x) * log(abs(x)+1, 1.1))
    data['count'] = data['count'].apply(lambda x: log(x+1, 1.1))

    dummy_cols = [
        'alerttype', 'devicetype',
        'devicevendor_code',
        # 'srcip', 'dstip',
        'srcipcategory', 'dstipcategory',
        'srcportcategory', 'dstportcategory',
    ]

    drop_cols = ['reportingdevice_code',  # 1889
                 #'srcip', 'dstip',

                 'protocol'  # 3271
                 ]

    data.drop(columns=drop_cols, inplace=True)

    uniq_val_dict = {}
    longest_uniq_val_list = 0
    for col in dummy_cols:
        uniq_vals = data[col].unique().tolist()
        if longest_uniq_val_list < len(uniq_vals):
            longest_uniq_val_list = len(uniq_vals)

        uniq_val_dict[col] = uniq_vals

    for col in dummy_cols:
        uniq_vals = uniq_val_dict[col]
        for i in range(longest_uniq_val_list - len(uniq_vals)):
            uniq_val_dict[col].append(uniq_val_dict[col][0])

    last_boundry = 0
    curr_id = ids[0]

    data_dict = {}

    for i, _id in enumerate(ids):

        if i % 10_000 == 0:
            print(f'{i}/{len(ids)}')

        if curr_id != _id:

            curr_id = _id

            data_part = data[last_boundry:i]
            data_part.pop('alert_ids')

            srcip_lists = [ip_to_bin_list(ip) for ip in data_part['srcip']]
            srcip_df = pd.DataFrame(srcip_lists, columns=[f'srcip_{i}' for i in range(40)])
            srcip_df.index = data_part.index
            data_part = pd.concat([data_part, srcip_df], axis=1)
            del srcip_lists
            del srcip_df

            dstip_lists = [ip_to_bin_list(ip) for ip in data_part['dstip']]
            dstip_df = pd.DataFrame(dstip_lists, columns=[f'dstip_{i}' for i in range(40)])
            dstip_df.index = data_part.index
            data_part = pd.concat([data_part, dstip_df], axis=1)
            del dstip_lists
            del dstip_df

            data_part.drop(columns=['srcip', 'dstip'], inplace=True)

            srcport_lists = [port_to_bin_list(port) for port in data_part['srcport']]
            srcport_df = pd.DataFrame(srcport_lists, columns=[f'srcport_{i}' for i in range(16)])
            srcport_df.index = data_part.index
            data_part = pd.concat([data_part, srcport_df], axis=1)
            del srcport_lists
            del srcport_df

            dstport_lists = [port_to_bin_list(port) for port in data_part['dstport']]
            dstport_df = pd.DataFrame(dstport_lists, columns=[f'dstport_{i}' for i in range(16)])
            dstport_df.index = data_part.index
            data_part = pd.concat([data_part, dstport_df], axis=1)
            del dstport_lists
            del dstport_df

            data_part.drop(columns=['srcport', 'dstport'], inplace=True)

            data_ext = pd.DataFrame(uniq_val_dict)

            data_part = pd.concat([data_part, data_ext])

            data_part = pd.get_dummies(data_part, columns=dummy_cols, prefix_sep='$', dummy_na=True)
            data_part = data_part[:-longest_uniq_val_list]

            last_boundry = i

            id_num_list = id_dict[id_dict['id'] == _id]['id_num'].to_list()
            if len(id_num_list) == 0:
                continue

            id_num = id_num_list[0]

            data_dict[id_num] = data_part.astype('int8')

    return data_dict
