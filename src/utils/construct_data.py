import pickle, json, decimal, math
import os

TRAIN_FILE = 'fold%s_train.json'
TEST_FILE  = 'fold%s_test.json'

rel_distance = 4     # the relative distance between clause
data_path = 'pair_data_4' # the output path

def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js

def write_json(data, file_path):
    # data_str = json.dumps(data)

    with open(file_path, 'w') as fw:
        json.dump(data, fw, ensure_ascii=False)

def construct_data(fold_id, data_type='train'):
    if data_type =='train':
        file = TRAIN_FILE % fold_id
    else:
        file = TEST_FILE % fold_id

    read_file_path = os.path.join('./data/split10', file)
    write_file_path = os.path.join('./data', data_path, file)
    if not os.path.exists(os.path.join('./data', data_path)):
        os.makedirs(os.path.join('./data', data_path))

    data_list = read_json(read_file_path)
    '''
    {"doc_id": id,
     "doc_len": len,
     "pairs"[[idi, idj]]:,
    "clauses": [{"clause_id": "10", "emotion_category": "happiness", "emotion_token": "兴奋", "clause": "黄某兴奋不已"}, {}]
    }
    '''
    data = []
    for doc in data_list:
        doc_id = doc['doc_id']
        doc_len = doc['doc_len']
        doc_pairs = doc['pairs']
        doc_clauses = doc['clauses']
        for i in range(doc_len):
            clause_1 = doc_clauses[i]
            for j in range(doc_len):
                if data_type =='test' and abs(i-j) <= rel_distance:
                    clause_2 = doc_clauses[j]
                    doc_data = {"doc_id": doc_id,
                                "doc_len": doc_len,
                                "pairs": doc_pairs,
                                "clauses": doc_clauses,
                                "clause_1": clause_1,
                                "clause_2": clause_2
                                }
                    data.append(doc_data)
                if data_type =='train' and abs(i-j) <= rel_distance:
                    clause_2 = doc_clauses[j]
                    doc_data = {"doc_id": doc_id,
                                "doc_len": doc_len,
                                "pairs": doc_pairs,
                                "clauses": doc_clauses,
                                "clause_1": clause_1,
                                "clause_2": clause_2
                                }
                    data.append(doc_data)
    print('The number of {} is: {}'.format(file, len(data)))

    write_json(data, write_file_path)



if __name__ == '__main__':
    n_fold = 10
    data_types = ['train', 'test']
    for fold_id in range(1,n_fold+1):
        for data_type in data_types:
            construct_data(fold_id, data_type)


'''
File: pair_data_2, 训练集为2，测试集为2
The number of fold1_train.json is: 122465
The number of fold1_test.json is: 9500
The number of fold2_train.json is: 121012
The number of fold2_test.json is: 10953
The number of fold3_train.json is: 121512
The number of fold3_test.json is: 10453
The number of fold4_train.json is: 119044
The number of fold4_test.json is: 12921
The number of fold5_train.json is: 116715
The number of fold5_test.json is: 15250
The number of fold6_train.json is: 118456
The number of fold6_test.json is: 13509
The number of fold7_train.json is: 119306
The number of fold7_test.json is: 12659
The number of fold8_train.json is: 114886
The number of fold8_test.json is: 17079
The number of fold9_train.json is: 116301
The number of fold9_test.json is: 15664
The number of fold10_train.json is: 117988
The number of fold10_test.json is: 13977
'''

'''
File: pair_data_3, 训练集为3，测试集为3
The number of fold1_train.json is: 165151
The number of fold1_test.json is: 12600
The number of fold2_train.json is: 163126
The number of fold2_test.json is: 14625
The number of fold3_train.json is: 163808
The number of fold3_test.json is: 13943
The number of fold4_train.json is: 160342
The number of fold4_test.json is: 17409
The number of fold5_train.json is: 157103
The number of fold5_test.json is: 20648
The number of fold6_train.json is: 159508
The number of fold6_test.json is: 18243
The number of fold7_train.json is: 160716
The number of fold7_test.json is: 17035
The number of fold8_train.json is: 154564
The number of fold8_test.json is: 23187
The number of fold9_train.json is: 156509
The number of fold9_test.json is: 21242
The number of fold10_train.json is: 158932
The number of fold10_test.json is: 18819
'''

'''
File: pair_data_4 训练集为4，测试集为4
The number of fold1_train.json is: 204345
The number of fold1_test.json is: 15320
The number of fold2_train.json is: 201762
The number of fold2_test.json is: 17903
The number of fold3_train.json is: 202616
The number of fold3_test.json is: 17049
The number of fold4_train.json is: 198146
The number of fold4_test.json is: 21519
The number of fold5_train.json is: 194009
The number of fold5_test.json is: 25656
The number of fold6_train.json is: 197060
The number of fold6_test.json is: 22605
The number of fold7_train.json is: 198636
The number of fold7_test.json is: 21029
The number of fold8_train.json is: 190764
The number of fold8_test.json is: 28901
The number of fold9_train.json is: 193227
The number of fold9_test.json is: 26438
The number of fold10_train.json is: 196420
The number of fold10_test.json is: 23245
'''

'''
File: pair_data_5 训练集为5，测试集为5
The number of fold1_train.json is: 240071
The number of fold1_test.json is: 17674
The number of fold2_train.json is: 236952
The number of fold2_test.json is: 20793
The number of fold3_train.json is: 237970
The number of fold3_test.json is: 19775
The number of fold4_train.json is: 232494
The number of fold4_test.json is: 25251
The number of fold5_train.json is: 227471
The number of fold5_test.json is: 30274
The number of fold6_train.json is: 231150
The number of fold6_test.json is: 26595
The number of fold7_train.json is: 233104
The number of fold7_test.json is: 24641
The number of fold8_train.json is: 223512
The number of fold8_test.json is: 34233
The number of fold9_train.json is: 226493
The number of fold9_test.json is: 31252
The number of fold10_train.json is: 230488
The number of fold10_test.json is: 27257
'''

'''
File: pair_data_6, 训练集为6，测试集为6
The number of fold1_train.json is: 272345
The number of fold1_test.json is: 19684
The number of fold2_train.json is: 268728
The number of fold2_test.json is: 23301
The number of fold3_train.json is: 269904
The number of fold3_test.json is: 22125
The number of fold4_train.json is: 263424
The number of fold4_test.json is: 28605
The number of fold5_train.json is: 257527
The number of fold5_test.json is: 34502
The number of fold6_train.json is: 261816
The number of fold6_test.json is: 30213
The number of fold7_train.json is: 264158
The number of fold7_test.json is: 27871
The number of fold8_train.json is: 252840
The number of fold8_test.json is: 39189
The number of fold9_train.json is: 256345
The number of fold9_test.json is: 35684
The number of fold10_train.json is: 261174
The number of fold10_test.json is: 30855
'''

'''
File: pair_data_7, 训练集为7，测试集为7
The number of fold1_train.json is: 301211
The number of fold1_test.json is: 21378
The number of fold2_train.json is: 297146
The number of fold2_test.json is: 25443
The number of fold3_train.json is: 298488
The number of fold3_test.json is: 24101
The number of fold4_train.json is: 291004
The number of fold4_test.json is: 31585
The number of fold5_train.json is: 284247
The number of fold5_test.json is: 38342
The number of fold6_train.json is: 289126
The number of fold6_test.json is: 33463
The number of fold7_train.json is: 291866
The number of fold7_test.json is: 30723
The number of fold8_train.json is: 278808
The number of fold8_test.json is: 43781
The number of fold9_train.json is: 282855
The number of fold9_test.json is: 39734
The number of fold10_train.json is: 288550
The number of fold10_test.json is: 34039
'''

'''
File: pair_data_8, 训练集为8，测试集为8
The number of fold1_train.json is: 326733
The number of fold1_test.json is: 22786
The number of fold2_train.json is: 322284
The number of fold2_test.json is: 27235
The number of fold3_train.json is: 323798
The number of fold3_test.json is: 25721
The number of fold4_train.json is: 315324
The number of fold4_test.json is: 34195
The number of fold5_train.json is: 307721
The number of fold5_test.json is: 41798
The number of fold6_train.json is: 313172
The number of fold6_test.json is: 36347
The number of fold7_train.json is: 316318
The number of fold7_test.json is: 33201
The number of fold8_train.json is: 301496
The number of fold8_test.json is: 48023
The number of fold9_train.json is: 306117
The number of fold9_test.json is: 43402
The number of fold10_train.json is: 312708
The number of fold10_test.json is: 36811
'''

'''
File: pair_data_9, 训练集为9，测试集为9
The number of fold1_train.json is: 349011
The number of fold1_test.json is: 23954
The number of fold2_train.json is: 344270
The number of fold2_test.json is: 28695
The number of fold3_train.json is: 345954
The number of fold3_test.json is: 27011
The number of fold4_train.json is: 336518
The number of fold4_test.json is: 36447
The number of fold5_train.json is: 328091
The number of fold5_test.json is: 44874
The number of fold6_train.json is: 334094
The number of fold6_test.json is: 38871
The number of fold7_train.json is: 337656
The number of fold7_test.json is: 35309
The number of fold8_train.json is: 321032
The number of fold8_test.json is: 51933
The number of fold9_train.json is: 326275
The number of fold9_test.json is: 46690
The number of fold10_train.json is: 333784
The number of fold10_test.json is: 39181
'''

'''
File: pair_data_10, 训练集为10，测试集为10
The number of fold1_train.json is: 368185
The number of fold1_test.json is: 24914
The number of fold2_train.json is: 363244
The number of fold2_test.json is: 29855
The number of fold3_train.json is: 365098
The number of fold3_test.json is: 28001
The number of fold4_train.json is: 354746
The number of fold4_test.json is: 38353
The number of fold5_train.json is: 345523
The number of fold5_test.json is: 47576
The number of fold6_train.json is: 352054
The number of fold6_test.json is: 41045
The number of fold7_train.json is: 356034
The number of fold7_test.json is: 37065
The number of fold8_train.json is: 337582
The number of fold8_test.json is: 55517
The number of fold9_train.json is: 343489
The number of fold9_test.json is: 49610
The number of fold10_train.json is: 351936
The number of fold10_test.json is: 41163
'''

'''
File: pair_data_11, 训练集为11，测试集为11
The number of fold1_train.json is: 384467
The number of fold1_test.json is: 25704
The number of fold2_train.json is: 379426
The number of fold2_test.json is: 30745
The number of fold3_train.json is: 381436
The number of fold3_test.json is: 28735
The number of fold4_train.json is: 370238
The number of fold4_test.json is: 39933
The number of fold5_train.json is: 360245
The number of fold5_test.json is: 49926
The number of fold6_train.json is: 367286
The number of fold6_test.json is: 42885
The number of fold7_train.json is: 371674
The number of fold7_test.json is: 38497
The number of fold8_train.json is: 351382
The number of fold8_test.json is: 58789
The number of fold9_train.json is: 358001
The number of fold9_test.json is: 52170
The number of fold10_train.json is: 367384
The number of fold10_test.json is: 42787
'''

'''
File: pair_data_12, 训练集为12，测试集为12
The number of fold1_train.json is: 398111
The number of fold1_test.json is: 26338
The number of fold2_train.json is: 393048
The number of fold2_test.json is: 31401
The number of fold3_train.json is: 395192
The number of fold3_test.json is: 29257
The number of fold4_train.json is: 383224
The number of fold4_test.json is: 41225
The number of fold5_train.json is: 372503
The number of fold5_test.json is: 51946
The number of fold6_train.json is: 380038
The number of fold6_test.json is: 44411
The number of fold7_train.json is: 384804
The number of fold7_test.json is: 39645
The number of fold8_train.json is: 362680
The number of fold8_test.json is: 61769
The number of fold9_train.json is: 370067
The number of fold9_test.json is: 54382
The number of fold10_train.json is: 380374
The number of fold10_test.json is: 44075
'''
