import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np
import random


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(input_size=config.embedding_dim, hidden_size=config.hidden_size+128, num_layers=config.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(config.hidden_size+128, len(config.ns))

    def forward(self, encoder_hidden, target):
        decoder_hidden = encoder_hidden
        batch_size = encoder_hidden.size(1)
        t = torch.empty([1, config.embedding_dim])
        t = t.unsqueeze(0)
        decoder_input = t.repeat(batch_size, 1, 1)

        decoder_outputs = torch.zeros([batch_size, config.max_len, len(config.ns)]).to(config.device)


        for t in range(config.max_len):
            decoder_output_t, decoder_hidden = self.forward_onestep(decoder_input,
                                                                    decoder_hidden)

            decoder_outputs[:, t, :] = decoder_output_t

            use_teacher_forcing = random.random() > 0.5
            if use_teacher_forcing:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                value, index = decoder_output_t.max(dim=-1)
                decoder_input = index.unsqueeze(dim=-1)

        return decoder_outputs, decoder_hidden


    def forward_onestep(self, decoder_input, pre_decoder_hidden):
        output, decoder_hidden = self.gru(decoder_input, pre_decoder_hidden)
        output_squeeze = output.squeeze(dim=1)
        output_fc = F.log_softmax(self.fc(output_squeeze),
                                  dim=-1)
        return output_fc, decoder_hidden


    def evaluate(self, encoder_hidden):
        decoder_hidden = encoder_hidden
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor([[config.ns.SOS]] * batch_size).to(config.device)
        decoder_outputs = torch.zeros([batch_size, config.max_len, len(config.ns)]).to(config.device)
        decoder_predict = []

        for t in range(config.max_len):
            decoder_output_t, decoder_hidden = self.forward_onestep(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t
            value, index = decoder_output_t.max(dim=-1)
            decoder_input = index.unsqueeze(dim=-1)
            decoder_predict.append(
                index.cpu().detach().numpy())

        decoder_predict = np.array(decoder_predict).transpose()
        return decoder_outputs, decoder_predict


if __name__ == '__main__':
    import dataset
    import config
    from encoder import Encoder

    data_loader = dataset.get_dataloader()
    encoder = Encoder()
    decoder = Decoder()
    for animation, input_api, output_api in data_loader:
        encoder_outputs, encoder_hidden = encoder(input)
        decoder_outputs, decoder_hidden = decoder(encoder_hidden, target)
        _, index = decoder_outputs.max(dim=-1)
        print('预测结果:', [config.ns.inverse_transform(one_stringNum.numpy()) for one_stringNum in index], '\n')
        print(decoder_outputs.size())  # decoder_outputs:[3, 10, 14]
        print(decoder_hidden.size())  # decoder_hidden:[1, 3, 4]
        break
