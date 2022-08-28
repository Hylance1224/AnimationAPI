from encoder import Encoder
from decoder import Decoder
import torch.nn as nn
import torch
import numpy as np


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, animation, input_api, target_api, CNN3D):
        encoder_outputs, encoder_hidden = self.encoder(input_api)
        CNN_features = CNN3D(animation, 'test')
        fusion_features = torch.cat((CNN_features, encoder_hidden), 2)
        decoder_outputs, decoder_hidden = self.decoder(fusion_features, target_api)
        return decoder_outputs, decoder_hidden

    def evaluate(self, input_api, input_api_length, input_video, CNN3D):
        encoder_outputs, encoder_hidden = self.encoder(input_api, input_api_length)
        CNN_features = CNN3D(input_video, 'test')
        fusion_features = torch.cat((CNN_features, encoder_hidden), 2)
        decoder_outputs, decoder_predict = self.decoder.evaluate(fusion_features)
        return decoder_outputs, decoder_predict
