import torch
from torch import nn, autograd
from _constraint_net import *

class Projection(nn.Module):
    def __init__(self, num_particles, dimension, constrains, num_iter = 3, stiffness = 1, boundary_nodes = None):
        super(Projection, self).__init__()
        self.num_iter = num_iter
        self.num_particles = num_particles
        self.dimension = dimension
        self.constrains = constrains
        self.boundary_nodes = boundary_nodes
        self.stiffness = stiffness
    
    def cal_delta_x(self, input_x):
        input_ = input_x.requires_grad_(True)
        output_ = self.constrains(input_)
        grad = autograd.grad(
            outputs=output_,
            inputs=input_,
            grad_outputs=torch.ones_like(output_),
            create_graph=True,
            retain_graph=True
        )[0]
        cons = output_
        # cons : B * 1; grad: B * num_particles * dimension
        eps = 1e-7  # avoid dividing by zero
        s = (cons.squeeze() / ((grad*grad).sum([1,2])+eps) ).expand(input_x.size()[1],input_x.size()[2],-1).permute(2,0,1)
        # delta_x = - ( c / sum(grad_x^2) ) * grad_x 
        return - s * grad 

    def cal_delta_x_boundary_nodes(self, input_x):
        input_ = input_x.requires_grad_(True)
        output_ = self.constrains(input_)
        grad = autograd.grad(
            outputs=output_,
            inputs=input_,
            grad_outputs=torch.ones_like(output_),
            create_graph=True,
            retain_graph=True
        )[0]            # grad: B * num_particles * dimension
        cons = output_  # cons : B * 1; 

        # set grad of boundary nodes to be zero
        grad[:,self.boundary_nodes,:] = 0
        eps = 1e-7  # avoid dividing by zero
        s = (cons.squeeze() / ((grad*grad).sum([1,2])+eps) ).expand(input_x.size()[1],input_x.size()[2],-1).permute(2,0,1)
        # delta_x = - ( c / sum(grad_x^2) ) * grad_x 
        return - s * grad 
    
    def cal_delta_x_soft(self, delta_x, i):
        # i: index of iteration
        k_iter = 1 - (1 - self.stiffness)**(1.0 / (i + 1))
        delta_x = delta_x * k_iter 
        return delta_x

    def get_delta_x(self, upd_x, i):
        if (not self.boundary_nodes): 
            delta_x = self.cal_delta_x(upd_x)  # calculate Delta_X  

        else: # process boundary nodes
            delta_x = self.cal_delta_x_boundary_nodes(upd_x)

        if (self.stiffness < 1):  # for soft projection
            delta_x = self.cal_delta_x_soft(delta_x, i)
        return delta_x

    def forward(self, x):
        # x : B * num_particles * dimension 
        upd_x = x
        for i in range(self.num_iter):
            delta_x = self.get_delta_x(upd_x, i)
            upd_x = upd_x + delta_x 
        return upd_x


class GroupProjection(nn.Module):
    # projs [Projection, ]
    # groups [[particle_idx, ], ] 
    def __init__(self, num_particles, dimension, projs, groups, num_iter = 3):
        super(GroupProjection, self).__init__()
        self.num_iter = num_iter
        self.num_particles = num_particles 
        self.dimension = dimension
        self.projs = projs
        self.groups = groups

    def forward(self, x):
        # x : B * num_particles * dimension 
        upd_x = x
        for ite in range(self.num_iter):
            delta_x = torch.zeros(x.shape)
            # form a larger batch containing each group
            # [batch, num_particles, dimension] -> [num_group*batch, num_particles_per_group, dimension]
            for i in range(len(self.projs)):
                proj = self.projs[i]
                groups = self.groups[i]
                group_x = torch.zeros([len(groups) * x.shape[0], proj.num_particles, x.shape[2]])
                for j in range(len(groups)):
                    group_idx = groups[j]
                    group_x[j*x.shape[0]:(j+1)*x.shape[0], :, :] = upd_x[:, group_idx, :]
                # Jacobi iteration (iter over each group)
                delta_x_group = proj.get_delta_x(group_x, ite)
                for j in range(len(groups)):
                    group_idx = groups[j]
                    delta_x[:, group_idx, :] += delta_x_group[j*x.shape[0]:(j+1)*x.shape[0], :, :]
            
            upd_x = upd_x + delta_x

        return upd_x

class GroupProjection2(nn.Module):
    # projs [Projection, ]
    # groups [[particle_idx, ], ] 
    def __init__(self, num_particles, dimension, projs, groups, num_iter = 3):
        super(GroupProjection2, self).__init__()
        self.num_iter = num_iter
        self.num_particles = num_particles 
        self.dimension = dimension
        self.projs = projs
        self.groups = groups

    def forward(self, x):
        upd_x = x
        for ite in range(self.num_iter):
            for i in range(len(self.projs)):
                proj = self.projs[i]
                groups = self.groups[i]
                for j in range(len(groups)):
                    group_idx = groups[j]
                    group_x = upd_x[:, group_idx, :]
                    upd_x[:, group_idx, :] = proj(group_x)
        return upd_x

