import torch.nn as nn
import config
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=config.embedding_dim,hidden_size=config.hidden_size,num_layers=config.num_layers,batch_first=True)


    def forward(self,input):
        output, hidden = self.gru(input)
        return output, hidden


if __name__=='__main__':
    import dataset
    data_loader = dataset.get_dataloader()
    encoder = Encoder()
    i = 0
    for animation, input_api in data_loader:
        encoder_outputs,encoder_hidden = encoder(input_api)
        print(encoder_outputs.size())
        print(encoder_hidden.size())
        i = i + 1
        if i == 5:
            break
