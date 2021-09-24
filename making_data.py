import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
import requests
import json

def remove_str_start_end(s, start, end):
    return s[:start] + s[end + 1:]

def remove_replace_str_start_end(s, start, end, value):
    return s[:start] + value + ' '+ s[end + 1:]

API_ENDPOINT = "https://www.wikidata.org/w/api.php"
def find_per(query):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': query
    }
    r = requests.get(API_ENDPOINT, params = params)
    try:
        return r.json()['search'][0]['id']
    except:
        return 'no_RE'

def return_wikidata(id_person):
    headers = {
        'Accept': 'application/json',
    }
    response = requests.get('http://www.wikidata.org/entity/' + str(id_person), headers=headers)
    return json.loads(response.text)['entities'][id_person]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='Train data')
    parser.add_argument('--test', dest='Test Data')
    parser.add_argument('--wikidata', dest = 'Wikidata file (pickle)')
    parser.add_argument('--outputname', dest = 'Output data')
    train_data = parser.train
    test_data = parser.test
    wiki_file = parser.wikidata
    outputname = parser.output

    with open(wiki_file, mode='rb') as f:
        wikidata = pickle.load(f)
    wikidata_dict = {}
    for i in range(len(wikidata)):
        wikidata_dict[wikidata[i][0]] = wikidata[i][1]

    tagger = SequenceTagger.load('ner')
    remove_sentences = []
    replacement_sentences= []
    USER_sentences = []
    USER_ner_sentences = []

    entity_info = []
    entity_data = []
    for i in tqdm(range(len(train_data))):
        try:
            sentence = Sentence(train_data['text'][i])
        except:
            remove_sentences.append('')
            replacement_sentences.append('')
            USER_sentences.append('')
            USER_ner_sentences.append('')
            continue
        tagger.predict(sentence)
        entities = reversed(sentence.get_spans('ner'))
        remove_sent = sentence.to_original_text()
        replace_sent= sentence.to_original_text()
        USER_sent =  sentence.to_original_text()
        USER_ner_sent = sentence.to_original_text()
        entity_data_ = []
        for entity in entities:
            remove_sent = remove_str_start_end(remove_sent, entity.start_pos, entity.end_pos)
            replace_sent = remove_replace_str_start_end(replace_sent, entity.start_pos, entity.end_pos, entity.labels[0].value)        
            entity_data_.append(entity)
            if 'PER' in entity.labels[0].value:
                user_id = find_per(entity.text)
                if user_id == 'no_RE':
                    replace_id = 'PER'
                else:
                    try:
                        replace_id = wikidata_dict[user_id]
                    except:
                        replace_id = 'PER'
                entity_info.append(replace_id)
                USER_sent = remove_replace_str_start_end(USER_sent, entity.start_pos, entity.end_pos, replace_id)
                USER_ner_sent = remove_replace_str_start_end(USER_ner_sent, entity.start_pos, entity.end_pos, replace_id)
            else:
                entity_info.append(entity.labels[0].value)
                USER_ner_sent = remove_replace_str_start_end(USER_sent, entity.start_pos, entity.end_pos, entity.labels[0].value)
        remove_sentences.append(remove_sent)
        replacement_sentences.append(replace_sent)
        USER_sentences.append(USER_sent)
        USER_ner_sentences.append(USER_ner_sent)
        entity_info = list(set(entity_info))
        entity_data.append([sentence, entity_data_])

    train_data['remove_sentence'] = remove_sentences
    train_data['replace'] = replacement_sentences
    train_data['USER_s'] = USER_sentences
    train_data['USER_s_ner'] = USER_ner_sentences
    train_data.to_csv(outputname + '_train.csv')
    with open(outputname + "_train_nerList.pickle", mode='wb') as f:
        pickle.dump([entity_info, entity_data], f)


    remove_sentences = []
    replacement_sentences= []
    USER_sentences = []
    USER_ner_sentences = []

    entity_info = []
    entity_data = []
    for i in tqdm(range(len(test_data))):
        try:
            sentence = Sentence(test_data['text'][i])
        except:
            remove_sentences.append('')
            replacement_sentences.append('')
            USER_sentences.append('')
            USER_ner_sentences.append('')
            continue
        tagger.predict(sentence)
        entities = reversed(sentence.get_spans('ner'))
        remove_sent = sentence.to_original_text()
        replace_sent= sentence.to_original_text()
        USER_sent =  sentence.to_original_text()
        USER_ner_sent = sentence.to_original_text()
        entity_data_ = []
        for entity in entities:
            remove_sent = remove_str_start_end(remove_sent, entity.start_pos, entity.end_pos)
            replace_sent = remove_replace_str_start_end(replace_sent, entity.start_pos, entity.end_pos, entity.labels[0].value)        
            entity_data_.append(entity)
            if 'PER' in entity.labels[0].value:
                user_id = find_per(entity.text)
                if user_id == 'no_RE':
                    replace_id = 'PER'
                else:
                    try:
                        replace_id = wikidata_dict[user_id]
                    except:
                        replace_id = 'PER'
                entity_info.append(replace_id)
                USER_sent = remove_replace_str_start_end(USER_sent, entity.start_pos, entity.end_pos, replace_id)
                USER_ner_sent = remove_replace_str_start_end(USER_ner_sent, entity.start_pos, entity.end_pos, replace_id)
            else:
                entity_info.append(entity.labels[0].value)
                USER_ner_sent = remove_replace_str_start_end(USER_sent, entity.start_pos, entity.end_pos, entity.labels[0].value)
        remove_sentences.append(remove_sent)
        replacement_sentences.append(replace_sent)
        USER_sentences.append(USER_sent)
        USER_ner_sentences.append(USER_ner_sent)
        entity_info = list(set(entity_info))
        entity_data.append([sentence, entity_data_])

    test_data['remove_sentence'] = remove_sentences
    test_data['replace'] = replacement_sentences
    test_data['USER_s'] = USER_sentences
    test_data['USER_s_ner'] = USER_ner_sentences

    test_data.to_csv(outputname + '_test.csv')
    with open(outputname + "_test_nerList.pickle", mode='wb') as f:
        pickle.dump([entity_info, entity_data], f)

if __name__ == '__main__':
    main()
