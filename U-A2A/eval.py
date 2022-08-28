import torch
from seq2seq import Seq2Seq
import config
from dataset import get_dataloader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def eval():
    seq2seq_model = Seq2Seq().to(config.device)
    seq2seq_model.load_state_dict(torch.load('./model/Seq2Seq_test/seq2seq_model.pkl'))
    loss_list = []
    acc_list = []
    data_loader = get_dataloader(train=False)
    with torch.no_grad():
        data_loader = get_dataloader(train=True)
        bar = tqdm(data_loader, total=len(data_loader), ascii=True, desc='test')
        for idx, (input, target, input_length, target_length) in enumerate(bar):
            input = input.to(config.device)
            target = target.to(config.device)
            input_length = input_length.to(config.device)

            decoder_ouputs, decoder_predict = seq2seq_model.evaluate(input, input_length)

            loss = F.nll_loss(decoder_ouputs.view(-1, len(config.ns)), target.view(-1),
                              ignore_index=config.ns.PAD)
            loss_list.append(loss.item())

            target_inverse_transformed = [config.ns.inverse_transform(i) for i in target.numpy()]
            predict_inverse_transformed = [config.ns.inverse_transform(i) for i in decoder_predict]

            cur_eq = [1 if target_inverse_transformed[i] == predict_inverse_transformed[i] else 0 for i in
                      range(len(target_inverse_transformed))]
            acc_list.extend(cur_eq)


            bar.set_description("mean acc:{:.6f} mean loss:{:.6f}".format(np.mean(acc_list), np.mean(loss_list)))


def interface(_input):
    seq2seq_model = Seq2Seq().to(config.device)
    seq2seq_model.load_state_dict(torch.load('./model/Seq2Seq_test/seq2seq_model.pkl'))
    input = list(_input)
    input_length = torch.LongTensor([len(input)])
    input = torch.LongTensor([config.ns.transform(input, max_len=config.max_len, add_eos=True)])
    with torch.no_grad():
        input = input.to(config.device)
        input_length = input_length.to(config.device)
        _, decoder_predict = seq2seq_model.evaluate(input, input_length)
        predict = [config.ns.inverse_transform(idx) for idx in decoder_predict]
        print(_input, '————>', predict[0])

1
if __name__ == "__main__":
    for i in range(3):
        str = input()
        interface(str)
