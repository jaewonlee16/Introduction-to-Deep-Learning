from glob import glob
import string

import torch
from torch.utils.data import Dataset, DataLoader, random_split

MAX_VOCAB_SIZE = 10_000 # must be lower than 89,527

# Text preprocessing; string to list
def preprocessing(text):
    text = text.replace("<br />", "")
    for p in string.punctuation:
        if p == "?": text = text.replace(p, " ? ")
        elif p == "!": text = text.replace(p, " ! ")
        elif p == "-" or p == "'": continue
        else: text = text.replace(p, " ")
    return text.strip().lower().split() 

# Customized Dataset and DataLoader
class ImdbDataset(Dataset):
    def __init__(self, text, label, tokenizer):
        self.text = list(map(tokenizer, text))
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        text, label = self.text[idx], self.label[idx]
        return torch.tensor(text, dtype = torch.int32), label

class ImdbDataloader(DataLoader):
    def __init__(self, dataset, batch_size, pad_first, shuffle = True):
        super().__init__(dataset = dataset,
                         batch_size = batch_size,
                         shuffle = shuffle,
                         collate_fn = self._make_batch)
        self.pad_first = pad_first

    # Batch generating function
    def _make_batch(self, samples):
        max_len = 0
        text_batch, label_batch, mask_batch = [], [], []
        for text, label in samples:
            if len(text) > max_len: max_len = len(text)
            label_batch.append(label)
        for text, _ in samples:
            padding = torch.zeros(max_len - len(text), dtype = torch.int32)
            if self.pad_first:
                text_batch.append(torch.cat((padding, text)))
                mask_batch.append(torch.cat((padding, torch.ones_like(text))))
            else:
                text_batch.append(torch.cat((text, padding)))
                mask_batch.append(torch.cat((torch.ones_like(text), padding)) == 0)
        return torch.stack(text_batch), torch.LongTensor(label_batch), torch.stack(mask_batch)

class Imdb:
    def __init__(self, max_vocab_size = MAX_VOCAB_SIZE):
        with open("./aclImdb/imdb.vocab", "r") as file:
            self._vocabs = {
                "<PAD>": 0,
                "<OOV>": 1,
                "<SOS>": 2,
                "<EOS>": 3
            }

            num_special_tokens = len(self._vocabs)

            for i in range(num_special_tokens, max_vocab_size + num_special_tokens):
                self._vocabs[file.readline().replace("\n", "")] = i

        path_format = "./aclImdb/{}/{}/*.txt"
        self._pos_text, self._neg_text = [], []

        for d in ["train", "test"]:
                for filename in glob(path_format.format(d, "pos")):
                    with open(filename, "r") as file:
                        self._pos_text.append(file.readline())
                for filename in glob(path_format.format(d, "neg")):
                    with open(filename, "r") as file:
                        self._neg_text.append(file.readline())

        self._pos_label = [1 for _ in self._pos_text]
        self._neg_label = [0 for _ in self._neg_text]

    def tokenizer(self, text):
        preprocessed_text = preprocessing(text)
        tokenized_text = [self._vocabs["<OOV>"] for _ in preprocessed_text]
        for i, word in enumerate(preprocessed_text):
            try: tokenized_text[i] = self._vocabs[word]
            except: pass
        return [self._vocabs["<SOS>"]] + tokenized_text + [self._vocabs["<EOS>"]]

    def make_loaders(self, batch_size = 2, pad_first = True):
        base = ImdbDataset(text = self._pos_text + self._neg_text,
                           label = self._pos_label + self._neg_label,
                           tokenizer = self.tokenizer)
        train, val, test = random_split(base, [30000, 10000, 10000])

        train_loader = ImdbDataloader(train, batch_size = batch_size, pad_first = pad_first)
        val_loader = ImdbDataloader(val, batch_size = batch_size, pad_first = pad_first)
        test_loader = ImdbDataloader(test, batch_size = batch_size, pad_first = pad_first)

        return (train_loader, val_loader, test_loader)
    
    def make_small_loaders(self, loader_size = 8, batch_size = 8, pad_first = True):
        base = ImdbDataset(text = self._pos_text[:loader_size] + self._neg_text[:loader_size],
                           label = self._pos_label[:loader_size] + self._neg_label[:loader_size],
                           tokenizer = self.tokenizer)
        train, val, test = random_split(base, [loader_size, loader_size // 2, loader_size // 2])

        train_loader = ImdbDataloader(train, batch_size = batch_size, pad_first = pad_first)
        # val_loader = ImdbDataloader(val, batch_size = batch_size, pad_first = pad_first)
        # test_loader = ImdbDataloader(test, batch_size = batch_size, pad_first = pad_first)

        # return (train_loader, val_loader, test_loader)
        return (train_loader, train_loader, train_loader)

    @property
    def vocabs(self):
        return self._vocabs
    
    @property
    def num_tokens(self):
        return len(self._vocabs)
    
    @property
    def pos_text(self):
        return self._pos_text

    @property
    def neg_text(self):
        return self._neg_text
