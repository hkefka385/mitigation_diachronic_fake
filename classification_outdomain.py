import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
import numpy as np

def evaluate(test_loader, model, device):
    model.eval()
    cnt = 0
    all_cnt = 0
    for batch in test_loader:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                }
        outputs= model(**inputs)
        correct = torch.argmax(outputs[0], axis = 1) == batch[2]
        for j in range(len(correct)):
            all_cnt += 1
            if correct[j] == True:
                cnt += 1
    print('Accurate: {}'.format(cnt / all_cnt))

def main():
    # seeds = 20
    # torch.manual_seed(seeds)
    # torch.cuda.manual_seed(seeds)
    # np.random.seed(seeds)
    # random.seed(seeds)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', dest='Pretarained filename')
    parser.add_argument('--test', dest='Test Data')
    pretrained_dir = parser.pretrained
    test_data = parser.test

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= True)
    model = BertForSequenceClassification.from_pretrained(pretrained_dir)
    tokenizer = tokenizer.from_pretrained(pretrained_dir)

    data_test = tokenizer.batch_encode_plus(
        test_data['text'].values,
        add_special_tokens=True,
        return_attention_mask = True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors = 'pt'
    )
    input_ids_test = data_test['input_ids']
    attention_masks_test = data_test['attention_mask']
    labels_test = torch.tensor(np.array(test_data.label.values, dtype=int))
    batch_size = 16
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    test_loader = DataLoader(
        dataset_test,
        sampler=RandomSampler(dataset_test),
        batch_size = batch_size
    )
    optim = AdamW(model.parameters(), lr = 1e-5)
    device = torch.device('')
    model.to(device)

    evaluate(test_loader, model, device)

if __name__ == '__main__':
    main()
