from torch import nn, Tensor

# num_features = [self.num*self.dim, 256, 256, 256, 256, 1]
class MLP_Constraint(nn.Module):
    def __init__(self, num_particles, dimension, num_features):
        super(MLP_Constraint, self).__init__()
        self.num = num_particles
        self.dim = dimension
        self.net = nn.ModuleDict()
        num_features = [self.num*self.dim] + num_features
        self.depth = len(num_features)-1
        for i in range(self.depth - 1):
            self.net['fc'+str(i+1)] = nn.Linear(in_features = num_features[i], out_features = num_features[i+1], bias = True)
            self.net['acti'+str(i+1)] = nn.LeakyReLU() #ReLU()
        i = self.depth - 1
        self.net['fc'+str(i+1)] = nn.Linear(in_features = num_features[i], out_features = num_features[i+1], bias = True)

    def forward(self, x):
        # x : B * m * n; out : B * (m*n)
        out = x.view([x.size()[0], x.size()[1]*x.size()[2]]) 
        for i in range(self.depth - 1):
            out = self.net['fc'+str(i+1)](out)
            out = self.net['acti'+str(i+1)](out) 
        out = self.net['fc'+str(self.depth)](out)
        return out # B * 1 