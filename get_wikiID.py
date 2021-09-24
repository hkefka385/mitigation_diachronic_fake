import bz2
import json
import pandas as pd
import pydash
import pickle
import argparse

def wikidata(filename):
    with bz2.open(filename, mode='rt') as f:
        f.read(2) # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue

def wikipID_get(filename):
    i = 0
    df_record_all = []
    for record in wikidata(filename):
        if pydash.has(record, 'claims.P39'):
            #print('i = '+str(i)+' item '+record['id']+'  started!'+'\n')
            user_id = pydash.get(record, 'claims.P39[0].mainsnak.datavalue.value.id')
            item_id = pydash.get(record, 'id')
            
            df_record = [item_id, user_id]
            df_record_all.append(df_record)
            i += 1
        elif pydash.has(record, 'claims.P106'):
            #print('i = '+str(i)+' item '+record['id']+'  started!'+'\n')
            user_id = pydash.get(record, 'claims.P106[0].mainsnak.datavalue.value.id')
            item_id = pydash.get(record, 'id')
            df_record = [item_id, user_id]
            df_record_all.append(df_record)
            i += 1
        else:
            continue
        #if i % 10000 == 0:
        #    print(i)
        if (i % 100000 == 0):
            with open(record['id'] + '_item.pickle', mode='wb') as f:
                pickle.dump(df_record_all, f)
            # print('i = '+str(i)+' item '+record['id']+'  started!'+'\n')
        else:
            continue
    return df_record_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', dest='filename')
    filename = parser.f
    record_all = wikipID_get(filename)
    
    with open(filename+ '_id.pickle', mode='wb') as f:
        pickle.dump(record_all, f)

if __name__ == '__main__':
    main()