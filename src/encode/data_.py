import csv
import collections
import operator
from csv import DictReader
from datetime import datetime
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import random
import numpy as np

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

'''
 只取训练集做测试
'''

def data_to_csv(datapath, is_to_csv):
    file_name = 'train.log.txt'
    data_path = datapath
    if is_to_csv:
        print('###### to csv.file ######\n')
        # 训练数据27个特征
        with open(data_path + 'train.all.origin.csv', 'w', newline='') as csvfile: # newline防止每两行就空一行
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
        with open(data_path + 'test.ctr.all.csv', 'w', newline='') as csvfile:  # newline防止每两行就空一行
            spamwriter = csv.writer(csvfile, dialect='excel')  # 读要转换的txt文件，文件每行各词间以@@@字符分隔
            with open(data_path + file_name, 'r') as filein:
                for i, line in enumerate(filein):
                    line_list = line.strip('\n').split('\t')
                    spamwriter.writerow(line_list)
        print('test-data读写完毕')

    file_name = 'train.all.origin.csv'
    data_path = datapath + file_name

    day_to_weekday = {4: '6', 5: '7', 6: '8', 0: '9', 1: '10', 2: '11', 3: '12'}
    train_data = pd.read_csv(data_path)
    train_data.iloc[:, 1] = train_data.iloc[:, 1].astype(int)
    if datapath.split('/')[-2] == '2259' or datapath.split('/')[-2] == '2997':
        train_len = len(train_data)
        train_split = int(train_len * 0.8)

        train_fm = train_data.iloc[:train_split, :]
        val_fm = train_data.iloc[train_split:, :]

        train_fm.to_csv(datapath + 'train.ctr.all.csv', index=None)
        val_fm.to_csv(datapath + 'val.ctr.all.csv', index=None)
    else:
        print('###### separate datas from train day ######\n')
        day_data_indexs = []
        for key in day_to_weekday.keys():
            day_datas = train_data[train_data.iloc[:, 1] == key]
            day_indexs = day_datas.index
            day_data_indexs.append([int(day_to_weekday[key]), int(day_indexs[0]), int(day_indexs[-1])])

        day_data_indexs_df = pd.DataFrame(data=day_data_indexs)
        day_data_indexs_df.to_csv(datapath + 'day_indexs.csv', index=None, header=None)

        day_indexs = np.array(day_data_indexs)
        train_indexs = day_indexs[day_indexs[:, 0] == 11][0]

        train_fm = train_data.iloc[:train_indexs[1], :]
        val_fm = train_data.iloc[train_indexs[1]:, :]

        train_fm.to_csv(datapath + 'train.ctr.all.csv', index=None)
        val_fm.to_csv(datapath + 'val.ctr.all.csv', index=None)

def to_libsvm_encode(datapath, sample_type):
    train_path = datapath + 'train.ctr.' + sample_type + '.csv'
    train_encode = datapath + 'train.ctr.' + sample_type + '.txt'
    val_path = datapath + 'val.ctr.all.csv'
    val_encode = datapath + 'val.ctr.' + sample_type + '.txt'
    test_path = datapath + 'test.ctr.all.csv'
    test_encode = datapath + 'test.ctr.' + sample_type + '.txt'
    feature_index = datapath + 'feat.ctr.' + sample_type + '.txt'

    field = ['hour', 'weekday', 'useragent', 'IP', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
             'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'advertiser']

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
                        if k == 'usertag':
                            v = '-'.join(v.split(',')[:3])
                        elif k == 'slotprice':
                            price = int(v)
                            if price > 100:
                                v = "101+"
                            elif price > 50:
                                v = "51-100"
                            elif price > 10:
                                v = "11-50"
                            elif price > 0:
                                v = "1-10"
                            else:
                                v = "0"
                        kv = k + '_' + v
                        features.append('{0}'.format(getIndices(kv)))
                        feature_indices.add(kv + '\t' + str(getIndices(kv)))
                    else:
                        kv = k + '_' + 'other'
                        features.append('{0}'.format(getIndices(kv)))

            if e % 100000 == 0:
                print(datetime.now(), 'creating train.txt...', e)

            outfile.write('{0},{1}\n'.format(row['click'], ','.join('{0}'.format(val) for val in features)))
    
    with open(val_encode, 'w') as outfile:
        for e, row in enumerate(DictReader(open(val_path)), start=1):
            features = []
            for k, v in row.items():
                if k in field:
                    if len(v) > 0:
                        if k == 'usertag':
                            v = '-'.join(v.split(',')[:3])
                        elif k == 'slotprice':
                            price = int(v)
                            if price > 100:
                                v = "101+"
                            elif price > 50:
                                v = "51-100"
                            elif price > 10:
                                v = "11-50"
                            elif price > 0:
                                v = "1-10"
                            else:
                                v = "0"
                        kv = k + '_' + v
                        indices = table.get(kv)
                        if indices is None:
                            kv = k + '_' + 'other'
                            features.append('{0}'.format(getIndices(kv)))
                        else:
                            features.append('{0}'.format(getIndices(kv)))
                    else:
                        kv = k + '_' + 'other'
                        features.append('{0}'.format(getIndices(kv)))

            if e % 100000 == 0:
                print(datetime.now(), 'creating val.txt...', e)

            outfile.write('{0},{1}\n'.format(row['click'], ','.join('{0}'.format(val) for val in features)))
    
    with open(test_encode, 'w') as outfile:
        for e, row in enumerate(DictReader(open(test_path)), start=1):
            features = []
            for k, v in row.items():
                if k in field:
                    if len(v) > 0:
                        if k == 'usertag':
                            v = '-'.join(v.split(',')[:3])
                        elif k == 'slotprice':
                            price = int(v)
                            if price > 100:
                                v = "101+"
                            elif price > 50:
                                v = "51-100"
                            elif price > 10:
                                v = "11-50"
                            elif price > 0:
                                v = "1-10"
                            else:
                                v = "0"
                        kv = k + '_' + v
                        indices = table.get(kv)
                        if indices is None:
                            kv = k + '_' + 'other'
                            features.append('{0}'.format(getIndices(kv)))
                        else:
                            features.append('{0}'.format(getIndices(kv)))
                    else:
                        kv = k + '_' + 'other'
                        features.append('{0}'.format(getIndices(kv)))

            if e % 100000 == 0:
                print(datetime.now(), 'creating test.txt...', e)

            outfile.write('{0},{1}\n'.format(row['click'], ','.join('{0}'.format(val) for val in features)))

    featvalue = sorted(table.items(), key=operator.itemgetter(1))
    fo = open(feature_index, 'w')
    fo.write(str(featvalue[-1][1] + 1) + '\n')
    for t, fv in enumerate(featvalue, start=1):
        if t > len(field):
            k = fv[0].split('_')[0]
            idx = field_index(k)
            fo.write(str(idx) + ':' + fv[0] + '\t' + str(fv[1]) + '\n')
        else:
            fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
    fo.close()


# def to_libsvm_encode(datapath, sample_type):
#     print('###### to libsvm encode ######\n')
#     oses = ["windows", "ios", "mac", "android", "linux"]
#     browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie"]
# 
#     f1s = ["weekday", "hour", "IP", "region", "city", "adexchange", "domain", "slotid", "slotwidth", "slotheight",
#            "slotvisibility", "slotformat", "creative", "advertiser"]
# 
#     f1sp = ["useragent", "slotprice"]
# 
#     f2s = ["weekday,region"]
# 
#     def featTrans(name, content):
#         content = content.lower()
#         if name == "useragent":
#             operation = "other"
#             for o in oses:
#                 if o in content:
#                     operation = o
#                     break
#             browser = "other"
#             for b in browsers:
#                 if b in content:
#                     browser = b
#                     break
#             return operation + "_" + browser
#         if name == "slotprice":
#             price = int(content)
#             if price > 100:
#                 return "101+"
#             elif price > 50:
#                 return "51-100"
#             elif price > 10:
#                 return "11-50"
#             elif price > 0:
#                 return "1-10"
#             else:
#                 return "0"
# 
#     def getTags(content):
#         if content == '\n' or len(content) == 0:
#             return ["null"]
#         return content.strip().split(',')
# 
#     # initialize
#     namecol = {}
#     featindex = {}
#     maxindex = 0
# 
#     fi = open(datapath + 'train.ctr.' + sample_type + '.csv', 'r')
# 
#     first = True
# 
#     featindex['truncate'] = maxindex
#     maxindex += 1
# 
#     for line in fi:
#         s = line.split(',')
#         if first:
#             first = False
#             for i in range(0, len(s)):
#                 namecol[s[i].strip()] = i
#                 if i > 0:
#                     featindex[str(i) + ':other'] = maxindex
#                     maxindex += 1
#             continue
#         for f in f1s:
#             col = namecol[f]
#             content = s[col]
#             feat = str(col) + ':' + content
#             if feat not in featindex:
#                 featindex[feat] = maxindex
#                 maxindex += 1
#         for f in f1sp:
#             col = namecol[f]
#             content = featTrans(f, s[col])
#             feat = str(col) + ':' + content
#             if feat not in featindex:
#                 featindex[feat] = maxindex
#                 maxindex += 1
#         col = namecol["usertag"]
#         tags = getTags(s[col])
#         for tag in tags:
#             feat = str(col) + ':' + tag
#             if feat not in featindex:
#                 featindex[feat] = maxindex
#                 maxindex += 1
# 
#     print('feature size: ' + str(maxindex))
#     featvalue = sorted(featindex.items(), key=operator.itemgetter(1))
# 
#     fo = open(datapath + 'feat.ctr.' + sample_type + '.txt', 'w')
#     fo.write(str(maxindex) + '\n')
#     for fv in featvalue:
#         fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
#     fo.close()
# 
#     # indexing train
#     print('indexing ' + datapath + 'train.ctr.' + sample_type + '.csv')
#     fi = open(datapath + 'train.ctr.' + sample_type + '.csv', 'r')
#     fo = open(datapath + 'train.ctr.' + sample_type + '.txt', 'w')
# 
#     first = True
#     for line in fi:
#         if first:
#             first = False
#             continue
#         s = line.split(',')
# 
#         fo.write(s[0])  # click
#         index = featindex['truncate']
#         fo.write(',' + str(index))
#         for f in f1s:  # every direct first order feature
#             col = namecol[f]
#             content = s[col]
#             feat = str(col) + ':' + content
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         for f in f1sp:
#             col = namecol[f]
#             content = featTrans(f, s[col])
#             feat = str(col) + ':' + content
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         col = namecol["usertag"]
#         tags = getTags(s[col])
#         for i, tag in enumerate(tags):
#             if i == 1:
#                 break
#             feat = str(col) + ':' + tag
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         fo.write('\n')
#     fo.close()
# 
#     # indexing val
#     print('indexing ' + datapath + 'val.ctr.all.csv')
#     fi = open(datapath + 'val.ctr.all.csv', 'r')
#     fo = open(datapath + 'val.ctr.' + sample_type + '.txt', 'w')
# 
#     first = True
#     for line in fi:
#         if first:
#             first = False
#             continue
#         s = line.split(',')
#         fo.write(s[0])  # click + winning price + hour + timestamp
#         index = featindex['truncate']
#         fo.write(',' + str(index))
#         for f in f1s:  # every direct first order feature
#             col = namecol[f]
#             if col >= len(s):
#                 print('col: ' + str(col))
#                 print(line)
#             content = s[col]
#             feat = str(col) + ':' + content
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         for f in f1sp:
#             col = namecol[f]
#             content = featTrans(f, s[col])
#             feat = str(col) + ':' + content
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         col = namecol["usertag"]
#         tags = getTags(s[col])
#         for i, tag in enumerate(tags):
#             if i == 1:
#                 break
#             feat = str(col) + ':' + tag
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         fo.write('\n')
# 
#     # indexing test
#     print('indexing ' + datapath + 'test.ctr.all.csv')
#     fi = open(datapath + 'test.ctr.all.csv', 'r')
#     fo = open(datapath + 'test.ctr.' + sample_type + '.txt', 'w')
# 
#     first = True
#     for line in fi:
#         if first:
#             first = False
#             continue
#         s = line.split(',')
#         fo.write(s[0])  # click + winning price + hour + timestamp
#         index = featindex['truncate']
#         fo.write(',' + str(index))
#         for f in f1s:  # every direct first order feature
#             col = namecol[f]
#             if col >= len(s):
#                 print('col: ' + str(col))
#                 print(line)
#             content = s[col]
#             feat = str(col) + ':' + content
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         for f in f1sp:
#             col = namecol[f]
#             content = featTrans(f, s[col])
#             feat = str(col) + ':' + content
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         col = namecol["usertag"]
#         tags = getTags(s[col])
#         for i, tag in enumerate(tags):
#             if i == 1:
#                 break
#             feat = str(col) + ':' + tag
#             if feat not in featindex:
#                 feat = str(col) + ':other'
#             index = featindex[feat]
#             fo.write(',' + str(index))
#         fo.write('\n')
#     fo.close()

def down_sample(data_path):
    # 负采样后达到的点击率
    CLICK_RATE = 0.001188  # 1:1000

    train_data = pd.read_csv(data_path + 'train.ctr.all.csv').values
    train_auc_num = len(train_data)

    click = np.sum(train_data[:, 0])
    total = train_auc_num
    train_sample_rate = click / (CLICK_RATE * (total - click))
    # 原始数据中的点击和曝光总数
    print('clicks: {0} impressions: {1}\n'.format(click, total))
    print('test_sample_rate is:', train_sample_rate)

    # 获取训练样本
    # test_sample_rate = test_sample_rate

    # 获取测试样本
    with open(data_path + 'train.ctr.down.csv', 'w') as fo:
        fi = open(data_path + 'train.ctr.all.csv')
        p = 0  # 原始正样本
        n = 0  # 原始负样本
        nn = 0  # 剩余的负样本
        c = 0  # 总数
        labels = 0
        for t, line in enumerate(fi, start=1):
            if t == 1:
                fo.write(line)
            else:
                c += 1
                label = line.split(',')[0]  # 是否点击标签
                if int(label) == 0:
                    n += 1
                    if random.randint(0, train_auc_num) <= train_auc_num * train_sample_rate:  # down sample, 选择对应数据量的负样本
                        fo.write(line)
                        nn += 1
                else:
                    p += 1
                    fo.write(line)

            if t % 10000 == 0:
                print(t)
        fi.close()
    print('数据负采样完成')


def rand_sample(data_path):
    train_data = pd.read_csv(data_path + 'train.ctr.all.csv')
    train_down_data = pd.read_csv(data_path + 'train.ctr.down.csv')

    sample_indexs = random.sample(range(len(train_data)), len(train_down_data))

    train_all_sample_data = train_data.iloc[sample_indexs, :]

    train_all_sample_data.to_csv(data_path + 'train.ctr.rand.csv', index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--campaign_id', default='3386/', help='1458, 3358, 3386, 3427, 3476')
    parser.add_argument('--is_to_csv', default=True)

    setup_seed(1)

    args = parser.parse_args()
    data_path = args.data_path + args.dataset_name + args.campaign_id

    if args.is_to_csv:
        data_to_csv(data_path, args.is_to_csv)

    to_libsvm_encode(data_path, 'all')

    # down denotes down sample, rand denotes random sample
    # down_sample(data_path)
    # to_libsvm_encode(data_path, 'down')
    #
    # rand_sample(data_path)
    # to_libsvm_encode(data_path, 'rand')





