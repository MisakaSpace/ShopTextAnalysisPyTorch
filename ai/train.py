import torch.utils.data
import json
from bpemb import BPEmb
import datetime
import torch.nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from ai.model import TextCNN, TextCNNDataset

dataset = TextCNNDataset(path='data/train.json', lang="uk", vs=200000)
model = TextCNN(output_size=dataset.get_label_size())

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

xdata = DataLoader(dataset)

# writer = SummaryWriter()

EPOCH = 750

for i in range(EPOCH):
    print(i, "start")
    for cls, data in xdata:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, cls)
        loss.backward()
        optimizer.step()
    print(i, loss)

    # writer.add_scalar('_CNN+Linear+WOHidden+Relu/Loss/train', loss.item(), i)

    torch.save(model, 'TCNN-E{}-L{}.pt'.format(i, loss.item()))
