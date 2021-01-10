import csv
import collections
import operator
from csv import DictReader
from datetime import datetime
import argparse
import pandas as pd

'''
 只取训练集做测试
'''

def data_to_csv(datapath, is_to_csv):
    file_name = 'train.log.txt'
    data_path = datapath
    if is_to_csv:
        print('###### to csv.file ######\n')
        # 训练数据27个特征
        with open(data_path + 'train.csv', 'w', newline='') as csvfile: # newline防止每两行就空一行
            spamwriter = csv.writer(csvfile, dialect='excel') # 读要转换的txt文件，文件每行各词间以@@@字符分隔
            with open(data_path + file_name, 'r') as filein:
                for i, line in enumerate(filein):
                    line_list = line.strip('\n').split('\t')
                    spamwriter.writerow(line_list)
        print('train-data读写完毕')

    file_name = 'test.log.txt'
    data_path = datapath
    if is_to_csv:
        print('###### to csv.file ######\n')
        # 训练数据27个特征
        with open(data_path + 'test.csv', 'w', newline='') as csvfile:  # newline防止每两行就空一行
            spamwriter = csv.writer(csvfile, dialect='excel')  # 读要转换的txt文件，文件每行各词间以@@@字符分隔
            with open(data_path + file_name, 'r') as filein:
                for i, line in enumerate(filein):
                    line_list = line.strip('\n').split('\t')
                    spamwriter.writerow(line_list)
        print('test-data读写完毕')

def separate_day_data(datapath, is_separate_data):
    file_name = 'train.csv'
    data_path = datapath + file_name
    if is_separate_data:
        day_to_weekday = {4: '6', 5: '7', 6: '8', 0: '9', 1: '10', 2: '11', 3: '12'}
        train_data = pd.read_csv(data_path, header=None).drop([0])
        train_data.iloc[:, 1] = train_data.iloc[:, 1].astype(int)
        print('###### separate datas from train day ######\n')
        day_data_indexs = []
        for key in day_to_weekday.keys():
            day_datas = train_data[train_data.iloc[:, 1] == key]
            day_indexs = day_datas.index
            day_data_indexs.append([int(day_to_weekday[key]), day_indexs[0] - 1, day_indexs[-1] - 1])

        day_data_indexs_df = pd.DataFrame(data=day_data_indexs)
        day_data_indexs_df.to_csv(datapath + 'train_day_index.csv', index=None, header=None)

def to_libsvm_encode(datapath):
    print('###### to libsvm encode ######\n')
    train_path = datapath+ 'train.csv'
    train_encode = datapath+ 'train.txt'
    test_path = datapath + 'test.csv'
    test_encode = datapath + 'test.txt'
    feature_index = datapath+ 'featindex.txt'

    field = ['hour', 'timestamp', 'useragent', 'IP', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
             'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'advertiser', 'nclick', 'nconversation']

    table = collections.defaultdict(lambda: 0)

    # 为特征名建立编号, filed
    def field_index(x):
        index = field.index(x)
        return index

    def getIndices(key):
        indices = table.get(key)
        if indices is None:
            indices = len(table)
            table[key] = indices
        return indices

    feature_indices = set()
    with open(train_encode, 'w') as outfile:
        for e, row in enumerate(DictReader(open(train_path)), start=1):
            features = []
            for k, v in row.items():
                if k in field:
                    if len(v) > 0:
                        kv = k + '_' + v
                        features.append('{0}'.format(getIndices(kv)))
                        feature_indices.add(kv + '\t' + str(getIndices(kv)))
                    else:
                        kv = k + '_' + 'other'
                        print(kv)
                        features.append('{0}'.format(getIndices(kv)))

            if e % 100000 == 0:
                print(datetime.now(), 'creating train.txt...', e)

            outfile.write('{0},{1}\n'.format(row['click'], ','.join('{0}'.format(val) for val in features)))

    with open(test_encode, 'w') as outfile:
        for e, row in enumerate(DictReader(open(test_path)), start=1):
            features = []
            for k, v in row.items():
                if k in field:
                    if len(v) > 0:
                        kv = k + '_' + v
                        features.append('{0}'.format(getIndices(kv)))
                        feature_indices.add(kv + '\t' + str(getIndices(kv)))
                    else:
                        kv = k + '_' + 'other'
                        print(kv)
                        features.append('{0}'.format(getIndices(kv)))

            if e % 100000 == 0:
                print(datetime.now(), 'creating test.txt...', e)

            outfile.write('{0},{1}\n'.format(row['click'], ','.join('{0}'.format(val) for val in features)))

    featvalue = sorted(table.items(), key=operator.itemgetter(1))
    fo = open(feature_index, 'w')
    for t, fv in enumerate(featvalue, start=1):
        if t > len(field):
            k = fv[0].split('_')[0]
            idx = field_index(k)
            fo.write(str(idx) + ':' + fv[0] + '\t' + str(fv[1]) + '\n')
        else:
            fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
    fo.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 2259, 3386')
    parser.add_argument('--is_to_csv', default=True)
    parser.add_argument('--is_separate_data', default=True)

    args = parser.parse_args()

    data_path = args.data_path + args.dataset_name + args.campaign_id
    data_to_csv(data_path, args.is_to_csv)

    separate_day_data(data_path, args.is_separate_data)

    to_libsvm_encode(data_path)

