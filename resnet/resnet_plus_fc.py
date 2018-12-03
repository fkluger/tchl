from torchvision.models import resnet
from torch import nn, stack
from torch.utils import model_zoo


class ResNetPlusLSTM(resnet.ResNet):

    def __init__(self, block, layers):

        super().__init__(block, layers)

        self.fc = None

        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=512 * block.expansion, hidden_size=512 * block.expansion, num_layers=1,
                            batch_first=True)

        self.fc_o = nn.Linear(512 * block.expansion, 1)
        self.fc_a = nn.Linear(512 * block.expansion, 1)

        self.avgpool = nn.AvgPool2d((12, 40), stride=1)

    def forward(self, x):

        seq_length = x.size()[1]

        resnet_outs = []
        for si in range(seq_length):
            y = x[:,si,:,:,:]

            y = self.conv1(y)
            y = self.bn1(y)
            y = self.relu(y)
            y = self.maxpool(y)

            y = self.layer1(y)
            y = self.layer2(y)
            y = self.layer3(y)
            y = self.layer4(y)

            # print("y: ", y.size())
            y = self.avgpool(y)
            y = y.view(y.size(0), -1)
            # print("yf: ", y.size())

            resnet_outs.append(y)

        x = stack(resnet_outs, dim=1)


        x, _ = self.lstm(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        return offset, angle


def resnet18rnn():
    model = ResNetPlusLSTM(resnet.BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']), strict=False)
    return model