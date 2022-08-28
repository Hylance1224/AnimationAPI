import torch
from torch import nn
from torch.autograd import Variable
from lib import ParseGRU


class ThreeD_conv(nn.Module):
    def __init__(self, opt, ndf=64, ngpu=1):
        super(ThreeD_conv, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.z_dim = opt.z_dim
        self.T = opt.T
        # self.image_size = opt.image_size
        self.image_height = opt.image_height
        self.image_width = opt.image_width
        self.n_channels = opt.n_channels
        # self.conv_size = int(opt.image_size/16)
        self.conv_size_height = int(opt.image_height / 16)
        self.conv_size_width = int(opt.image_width / 16)

        self.encoder = nn.Sequential(
            nn.Conv3d(opt.n_channels, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int((ndf*8)*(self.T/16)*self.conv_size_height*self.conv_size_width),self.z_dim ),#6*6
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.z_dim,int((ndf*8)*(self.T/16)*self.conv_size_height*self.conv_size_width)),#6*6
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d((ndf*8), ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf*4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 2, ndf , 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf , opt.n_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input, mode):
        bs = input.size(0)
        if mode == 'train':
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                feature = self.encoder(input)
                z = self.fc1(feature.view(bs, -1))
                feature = self.fc2(z).reshape(bs, self.ndf * 8, int(self.T / 16), self.conv_size_height,
                                              self.conv_size_width)
                output = self.decoder(feature).view(bs, self.n_channels, self.T, self.image_height, self.image_width)

            return output
        else:
            feature = self.encoder(input)
            z = self.fc1(feature.view(bs, -1))  # [batch_size, hidden_size]
            z1 = z.unsqueeze(0)  # [1, batch_size, hidden_size]
            return z1


if __name__ == '__main__':
    parse = ParseGRU()
    opt = parse.args
    checkpoint = torch.load('model/3DCNN/model8000')
    state_dict = checkpoint['model']
    CNN3D = ThreeD_conv(opt)
    CNN3D.load_state_dict(state_dict)

    import dataset
    data_loader = dataset.get_dataloader()
    i = 0
    for input in data_loader:
        print(input.size())
        features = CNN3D(input = input, mode='test')
        print(features.size())
        i = i + 1
        if i == 5:
            break

