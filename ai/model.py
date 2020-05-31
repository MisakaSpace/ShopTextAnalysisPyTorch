import torch.utils.data
import json
from bpemb import BPEmb
import torch.nn
import random


class TextCNNDataset(torch.utils.data.Dataset):
    def __init__(self, path, lang, vs) -> None:
        super().__init__()
        self._bpemb_uk = BPEmb(lang=lang, vs=vs)

        with open(path, encoding='utf-8') as f:
            data = json.load(f)

            self._labels = {value: number for number, value in enumerate(data.keys())}
            self._labels_r = {number: value for number, value in enumerate(data.keys())}
            self._data = []
            for key in data.keys():
                for intent in data[key]:
                    embed_data = self._bpemb_uk.embed(intent)
                    intent_tensor = torch.tensor(embed_data)
                    # output_tensor = torch.tensor([1. if key==key_pos else 0. for key_pos in data.keys()]).long()
                    output_tensor = torch.tensor(self._labels[key])
                    intent_tensor = self.upsize(intent_tensor, 10)
                    self._data.append((output_tensor, intent_tensor))
            random.shuffle(self._data)

    def __getitem__(self, index: int):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def embed(self, text):
        return self._bpemb_uk.embed(text)

    def get_label_size(self):
        return len(self._labels)

    def get_label_by_index(self, index):
        return self._labels_r[index]

    @staticmethod
    def upsize(input_tensor, outdim):
        current_dim = input_tensor.size()[0]
        if current_dim > outdim:
            return input_tensor[:outdim]
        empty_tensor = torch.tensor([[-1.] * 100])
        for _ in range(outdim - current_dim):
            input_tensor = torch.cat((input_tensor, empty_tensor))

        return input_tensor


class TextCNN(torch.nn.Module):
    """ Нейрона модель для класифікації тексту, яка базується на згортках"""

    def __init__(self, output_size):
        super().__init__()
        self.cn1_4 = torch.nn.Conv1d(10, 1, 4)
        self.cn1_3 = torch.nn.Conv1d(10, 1, 3)
        self.cn1_2 = torch.nn.Conv1d(10, 1, 2)
        self.ln1 = torch.nn.Linear(30, output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.ln1.weight.data.uniform_(-initrange, initrange)
        self.ln1.bias.data.zero_()
        self.cn1_4.weight.data.uniform_(-initrange, initrange)
        self.cn1_3.weight.data.uniform_(-initrange, initrange)
        self.cn1_2.weight.data.uniform_(-initrange, initrange)

    def forward(self, data):
        first_layer = [torch.nn.AdaptiveMaxPool1d(10)(torch.relu(self.cn1_4(data))),
                       torch.nn.AdaptiveMaxPool1d(10)(torch.relu(self.cn1_3(data))),
                       torch.nn.AdaptiveMaxPool1d(10)(torch.relu(self.cn1_2(data)))]
        x = torch.cat(first_layer, dim=1)
        cn1_res = x.view(-1, 30)
        return self.ln1(cn1_res).softmax(dim=1)