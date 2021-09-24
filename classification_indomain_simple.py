#from sklearn.model_selection import train_test_split
import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
import transformers
from tqdm import tqdm

seeds = 20
torch.manual_seed(seeds)
torch.cuda.manual_seed(seeds)
np.random.seed(seeds)
# random.seed(seeds)

def make_loader(train_data, test_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= True)
    data_train = tokenizer.batch_encode_plus(
        train_data['text'].values,
        add_special_tokens=True,
        return_attention_mask = True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors = 'pt'
    )

    data_test = tokenizer.batch_encode_plus(
        test_data['text'].values,
        add_special_tokens=True,
        return_attention_mask = True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors = 'pt'
    )
    input_ids_train = data_train['input_ids']
    attention_masks_train = data_train['attention_mask']
    labels_train = torch.tensor(np.array(train_data.label.values, dtype=int))

    input_ids_test = data_test['input_ids']
    attention_masks_test = data_test['attention_mask']
    labels_test = torch.tensor(np.array(test_data.label.values, dtype=int))
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    batch_size = 16

    train_loader = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size = batch_size
    )

    test_loader = DataLoader(
        dataset_test,
        sampler=RandomSampler(dataset_test),
        batch_size = batch_size
    )

    return train_loader, test_loader, tokenizer


def train(model, optim, device, epochs, train_loader):
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        loss_train_total = 0
        for batch in progress_bar:
            optim.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }
            outputs= model(**inputs)
            loss = outputs[0]
            loss_train_total = loss.item()
            loss.backward()
            optim.step()
            print(loss_train_total)
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        
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

def save(output_dir, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='Pretarained filename')
    parser.add_argument('--test', dest='Test Data')
    parser.add_argument('--n_epoch', dest = 'N of epoch')
    parser.add_argument('--output', dest = 'output folder')
    train_data = parser.train
    test_data = parser.test
    epochs = parser.n_epoch
    output_dir = parser.output

    train_loader, test_loader, tokenizer = make_loader(train_data, test_data)
    
    transformers.logging.set_verbosity(10)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, output_attentions=False, output_hidden_states=False)
    optim = AdamW(model.parameters(), lr = 1e-5)
    device = torch.device('')

    train(model, optim, device, epochs, train_loader)
    evaluate(test_loader, model, device)
    save(output_dir, model, tokenizer)


if __name__ == '__main__':
    main()
