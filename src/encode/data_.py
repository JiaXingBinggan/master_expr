import csv
import collections
import operator
from csv import DictReader
from datetime import datetime
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import random

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
        day_data_indexs_df.to_csv(datapath + 'day_indexs.csv', index=None, header=None)


def to_libsvm_encode(datapath):
    print('###### to libsvm encode ######\n')
    oses = ["windows", "ios", "mac", "android", "linux"]
    browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie"]

    f1s = ["weekday", "hour", "IP", "region", "city", "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser"]

    f1sp = ["useragent", "slotprice"]

    f2s = ["weekday,region"]

    def featTrans(name, content):
        content = content.lower()
        if name == "useragent":
            operation = "other"
            for o in oses:
                if o in content:
                    operation = o
                    break
            browser = "other"
            for b in browsers:
                if b in content:
                    browser = b
                    break
            return operation + "_" + browser
        if name == "slotprice":
            price = int(content)
            if price > 100:
                return "101+"
            elif price > 50:
                return "51-100"
            elif price > 10:
                return "11-50"
            elif price > 0:
                return "1-10"
            else:
                return "0"

    def getTags(content):
        if content == '\n' or len(content) == 0:
            return ["null"]
        return content.strip().split(',')[:5]

    # initialize
    namecol = {}
    featindex = {}
    maxindex = 0
    fi = open(datapath + 'train.csv', 'r')
    first = True

    featindex['truncate'] = maxindex
    maxindex += 1

    for line in fi:
        s = line.split(',')
        if first:
            first = False
            for i in range(0, len(s)):
                namecol[s[i].strip()] = i
                if i > 0:
                    featindex[str(i) + ':other'] = maxindex
                    maxindex += 1
            continue
        for f in f1s:
            col = namecol[f]
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                featindex[feat] = maxindex
                maxindex += 1
        for f in f1sp:
            col = namecol[f]
            content = featTrans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                featindex[feat] = maxindex
                maxindex += 1
        col = namecol["usertag"]
        tags = getTags(s[col])
        # for tag in tags:
        feat = str(col) + ':' + ''.join(tags)
        if feat not in featindex:
            featindex[feat] = maxindex
            maxindex += 1

    print('feature size: ' + str(maxindex))
    featvalue = sorted(featindex.items(), key=operator.itemgetter(1))
    fo = open(datapath + 'feat.ctr.txt', 'w')
    fo.write(str(maxindex) + '\n')
    for fv in featvalue:
        fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
    fo.close()

    # indexing train
    print('indexing ' + datapath + 'train.csv')
    fi = open(datapath + 'train.csv', 'r')
    fo = open(datapath + 'train.ctr.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        fo.write(s[0])  # click + winning price
        index = featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        for f in f1sp:
            col = namecol[f]
            content = featTrans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        col = namecol["usertag"]
        tags = getTags(s[col])
        # for tag in tags:
        feat = str(col) + ':' + ''.join(tags)
        if feat not in featindex:
            feat = str(col) + ':other'
        index = featindex[feat]
        fo.write(',' + str(index))
        fo.write('\n')
    fo.close()

    # indexing test
    print('indexing ' + datapath + 'test.csv')

    fi = open(datapath + 'test.csv', 'r')

    fo = open(datapath + 'test.ctr.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        fo.write(s[0])  # click + winning price
        index = featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            if col >= len(s):
                print('col: ' + str(col))
                print(line)
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        for f in f1sp:
            col = namecol[f]
            content = featTrans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        col = namecol["usertag"]
        tags = getTags(s[col])
        # for tag in tags:
        feat = str(col) + ':' + ''.join(tags)
        if feat not in featindex:
            feat = str(col) + ':other'
        index = featindex[feat]
        fo.write(',' + str(index))
        fo.write('\n')
    fo.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 2259, 3358, 3386, 3427, 3476')
    parser.add_argument('--is_to_csv', default=True)
    parser.add_argument('--is_separate_data', default=True)

    args = parser.parse_args()

    data_path = args.data_path + args.dataset_name + args.campaign_id

    data_to_csv(data_path, args.is_to_csv)

    separate_day_data(data_path, args.is_separate_data)

    to_libsvm_encode(data_path)



