#Author by Cqy2019
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import OrderedDict

class RetrievalModel(nn.Module):
    '''
    CNN for image retrieval
    '''
    def __init__(self, params, fine_tune_flag=True):
        super(RetrievalModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(in_features=512, out_features=params.code_size)
        self.tanh = nn.Tanh()
        for p in self.resnet.parameters():
            p.requires_grad = True

    def forward(self, image):
        out = self.resnet(image)
        out = self.tanh(out)

        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    model = RetrievalModel()
    a = torch.randn(1,3,224,224)
    print(a.size())
    out = model(a)
    print(out[0].size())
    for i in range(len(out[0])):
        if out[0][i] > 0:
            out[0][i] = 1
    print(out)