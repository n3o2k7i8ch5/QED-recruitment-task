from math import log

import pandas as pd

from ip_to_tensor import ip_to_bin_list


def prep_data(data: pd.DataFrame):
    dstip_lists = [ip_to_bin_list(ip) for ip in data['ip']]
    dstip_df = pd.DataFrame(dstip_lists, columns=[f'p_{i}' for i in range(40)])
    dstip_df.index = data.index
    data = pd.concat([data, dstip_df], axis=1)
    data.pop('ip')

    data['timestamp_dist'] = data['timestamp_dist'].apply(lambda x: log(x + 1, 2))

    data['correlatedcount'] = data['correlatedcount'].apply(lambda x: log(x + 1, 1.2))

    data['srcip_cd'] = data['srcip_cd'].apply(lambda x: log(x + 1, 1.2))
    data['dstip_cd'] = data['dstip_cd'].apply(lambda x: log(x + 1, 1.2))

    data['srcport_cd'] = data['srcport_cd'].apply(lambda x: log(x + 1, 1.2))
    data['dstport_cd'] = data['dstport_cd'].apply(lambda x: log(x + 1, 1.2))

    data['thrcnt_month'] = data['thrcnt_month'].apply(lambda x: log(x + 1, 1.2))
    data['thrcnt_week'] = data['thrcnt_week'].apply(lambda x: log(x + 1, 1.2))
    data['thrcnt_day'] = data['thrcnt_day'].apply(lambda x: log(x + 1, 1.2))

    data['alerttype_cd'] = data['alerttype_cd'].apply(lambda x: log(x + 1, 1.2))
    data['direction_cd'] = data['direction_cd'].apply(lambda x: log(x + 1, 1.2))
    data['eventname_cd'] = data['eventname_cd'].apply(lambda x: log(x + 1, 1.2))
    data['severity_cd'] = data['severity_cd'].apply(lambda x: log(x + 1, 1.2))
    data['reportingdevice_cd'] = data['reportingdevice_cd'].apply(lambda x: log(x + 1, 1.2))
    data['devicetype_cd'] = data['devicetype_cd'].apply(lambda x: log(x + 1, 1.2))
    data['devicevendor_cd'] = data['devicevendor_cd'].apply(lambda x: log(x + 1, 1.2))
    data['domain_cd'] = data['domain_cd'].apply(lambda x: log(x + 1, 1.2))
    data['protocol_cd'] = data['protocol_cd'].apply(lambda x: log(x + 1, 1.2))
    data['username_cd'] = data['username_cd'].apply(lambda x: log(x + 1, 1.2))
    data['srcipcategory_cd'] = data['srcipcategory_cd'].apply(lambda x: log(x + 1, 1.2))
    data['dstipcategory_cd'] = data['dstipcategory_cd'].apply(lambda x: log(x + 1, 1.2))

    data = pd.get_dummies(data, columns=[
        'categoryname', 'ipcategory_name', 'ipcategory_scope',
        'parent_category', 'grandparent_category', 'overallseverity', 'weekday',
        'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10',
        'score',

        #'alerttype_cd', 'direction_cd', 'eventname_cd', 'severity_cd',
        #'reportingdevice_cd', 'devicetype_cd', 'devicevendor_cd', 'domain_cd',
        #'protocol_cd', 'username_cd', 'srcipcategory_cd', 'dstipcategory_cd',
        'isiptrusted', 'untrustscore', 'flowscore', 'trustscore', 'enforcementscore',

        'dstipcategory_dominate', 'srcipcategory_dominate',
        'dstportcategory_dominate', 'srcportcategory_dominate',
        'p6', 'p9', 'p5m', 'p5w', 'p5d', 'p8m', 'p8w', 'p8d'
    ], prefix_sep='$', dummy_na=True)

    data.drop(columns=['start_hour', 'start_minute', 'start_second'], inplace=True)

    return data
