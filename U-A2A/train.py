from seq2seq import Seq2Seq
from torch.optim import Adam
from tqdm import tqdm
import os
import dataset
import config
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from eval import eval
from torchvision import datasets, transforms
from encoderCNN import ThreeD_conv
from lib import ParseGRU


seq2seq_model = Seq2Seq().to(config.device)
optimizer = Adam(seq2seq_model.parameters(), lr=0.001)
if os.path.exists('./model/Seq2Seq_test/seq2seq_model.pkl'):
    seq2seq_model.load_state_dict(torch.load('./model/Seq2Seq_test/seq2seq_model.pkl'))
    optimizer.load_state_dict(torch.load('./model/Seq2Seq_test/seq2seq_optimizer.pkl'))

loss_list = []

parse = ParseGRU()
opt = parse.args

checkpoint = torch.load('model/3DCNN/model8000')
state_dict = checkpoint['model']
CNN3D= ThreeD_conv(opt)
CNN3D.load_state_dict(state_dict)


def train(epoch):
    data_loader = dataset.get_dataloader(train=True)
    bar = tqdm(data_loader, total=len(data_loader), ascii=True, desc='train')
    for idx, (animation, input_api, target_api) in enumerate(bar):
        input_api = input_api.to(config.device)
        target_api = target_api.to(config.device)

        decoder_ouputs, _ = seq2seq_model(animation, input_api, target_api, CNN3D)

        decoder_ouputs = decoder_ouputs.view(-1, len(config.ns))  # [3 * 10, 14]
        target = target.view(-1)  # [3 * 10]

        loss = F.nll_loss(decoder_ouputs, target, ignore_index=config.ns.PAD)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        bar.set_description("epoch:{} idx:{} loss:{:.3f}".format(epoch, idx, np.mean(loss_list)))
        if not (idx % 100):
            torch.save(seq2seq_model.state_dict(), './model/Seq2Seq_test/seq2seq_model.pkl')
            torch.save(optimizer.state_dict(), './model/Seq2Seq_test/seq2seq_optimizer.pkl')


if __name__ == '__main__':
    for i in range(3):
        train(i)
        eval()

    plt.figure(figsize=(50, 8))
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
