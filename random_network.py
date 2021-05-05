from torch import nn

#TODO: experiment with network parameters. convolutional network (see https://github.com/openai/random-network-distillation, maybe follow the architecture of toad-gan for this part)
def create_random_network(input_size, output_size, device):

    class RandomNet(nn.Module):
        def __init__(self, device):
            super(RandomNet, self).__init__()
            self.device = device
            self.main = nn.Sequential(
                nn.Linear(in_features = input_size, out_features = output_size, bias = False),
            )
            self.norm = nn.BatchNorm1d(output_size, affine=False)

        def forward(self, input):
            #return self.norm(self.main(input).unsqueeze(0)).squeeze()
            return self.main(input)
            
    return RandomNet(device)
