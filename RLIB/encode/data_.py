import csv
import collections
import operator
from csv import DictReader
from datetime import datetime
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from itertools import islice
import random

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

    day_indexs = pd.read_csv(datapath + 'day_indexs.csv', header=None).values.astype(int)
    train_indexs = day_indexs[day_indexs[:, 0] == 11][0]
    test_indexs = day_indexs[day_indexs[:, 0] == 12][0]

    columns = list(islice(open(datapath + 'train.csv', 'r'), 0, 1))

    # initialize
    namecol = {}
    featindex = {}
    maxindex = 0
    fi = columns + list(islice(open(datapath + 'train.csv', 'r'), int(train_indexs[1]), int(train_indexs[2]) + 1))
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
    fo = open(datapath + 'feat.bid.txt', 'w')
    fo.write(str(maxindex) + '\n')
    for fv in featvalue:
        fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
    fo.close()

    # indexing train
    print('indexing ' + datapath + 'train.csv')
    fi = columns + list(islice(open(datapath + 'train.csv', 'r'), int(train_indexs[1]), int(train_indexs[2]) + 1))
    fo = open(datapath + 'train.bid.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + s[4][8: 12])  # click + winning price + hour + timestamp
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

    fi = columns + list(islice(open(datapath + 'train.csv', 'r'), int(test_indexs[1]), int(test_indexs[2]) + 1))

    fo = open(datapath + 'test.bid.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + s[4][8: 12])  # click + winning price + hour + timestamp
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
    parser.add_argument('--campaign_id', default='3386/', help='1458, 3358, 3386, 3427')
    parser.add_argument('--is_to_csv', default=True)
    parser.add_argument('--is_separate_data', default=True)

    args = parser.parse_args()

    data_path = args.data_path + args.dataset_name + args.campaign_id

    to_libsvm_encode(data_path)



