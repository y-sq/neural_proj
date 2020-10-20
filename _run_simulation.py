import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import path, mkdir
from PIL import Image, ImageDraw
import sys, glob, time

class PBD_Simulation():
    def __init__(self, init_pos, init_vel, proj, if_cuda = False):
        self.pos = init_pos
        self.vel = init_vel
        self.proj = proj
        self.before_proj_pos = init_pos
        self.if_cuda = if_cuda

    # externel_force: same dimension as pos/vel
    # dt: time stamp, which doesn't have to be fixed
    def advect(self, external_force, dt):
        self.vel = self.vel + external_force * dt
        self.before_proj_pos = self.pos + self.vel * dt

        data = self.before_proj_pos
        d = torch.Tensor(data[None, :, :]).cuda() if self.if_cuda else torch.Tensor(data[None, :, :])
        pred = self.proj(d)[0,:,:].detach().cpu().numpy()

        self.vel = (pred - self.pos) / dt
        self.pos = pred        

    def write_file(self, root_path, frame_id):
        if not path.exists(root_path):
                mkdir(root_path) 
        filename = root_path + "particles_frame_" + str(frame_id) + ".txt" 
        c = np.savetxt(filename, self.pos.ravel(), delimiter ='\n') 
    
    def draw_fig_2d(self, root_path, frame_id, xy_max, force = None):
        fig, ax = plt.subplots(figsize=(10,10))

        # draw force arrow            
        if (np.array(force).any()):
            X = []; Y = []; U = []; V = []
            for i in range(self.pos.shape[0]):
                if (np.linalg.norm(force[i])>0):
                    X.append(self.before_proj_pos[i][0])
                    Y.append(self.before_proj_pos[i][1])
                    U.append(force[i][0]) 
                    V.append(force[i][1]) 
            ax.quiver(X, Y, U, V, angles='xy', scale_units='xy')

        ax.scatter(self.before_proj_pos[:,0], self.before_proj_pos[:,1], c = 'y', label = "before projection")
        ax.scatter(self.pos[:,0], self.pos[:,1],  c = 'b', label = "after projection")       
        ax.grid()
        fig.legend()
        # ax.set(xlabel='X', ylabel='Y', title='yellow: before projection; blue: after projection.')
        ax.set_xlim(-xy_max, xy_max)
        ax.set_ylim(-xy_max, xy_max)
    
        if not path.exists(root_path):
            mkdir(root_path) 
        plt.savefig(root_path + "figure_frame_" + str(frame_id) + ".png" )


    def draw_fig_3d(self, root_path, frame_id, xy_max, force = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # draw force arrow            
        if (np.array(force).any()):
            X = []; Y = []; Z = []; U = []; V = []; W = []
            for i in range(self.pos.shape[0]):
                if (np.linalg.norm(force[i])>0):
                    X.append(self.before_proj_pos[i][0])
                    Y.append(self.before_proj_pos[i][1])
                    Z.append(self.before_proj_pos[i][2])
                    U.append(force[i][0]) 
                    V.append(force[i][1]) 
                    W.append(force[i][2]) 
            ax.quiver(X, Y, Z, U, V, W)

        ax.scatter(self.before_proj_pos[:,0], self.before_proj_pos[:,1], self.before_proj_pos[:,2], c = 'y', label = "before projection")
        ax.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2], c = 'b', label = "after projection")       
        ax.grid()
        fig.legend()
        # ax.set(xlabel='X', ylabel='Y', title='yellow: before projection; blue: after projection.')
        ax.set_xlim(-xy_max, xy_max)
        ax.set_ylim(-xy_max, xy_max)
        ax.set_zlim(-xy_max, xy_max)
    
        if not path.exists(root_path):
            mkdir(root_path) 
        plt.savefig(root_path + "figure_frame_" + str(frame_id) + ".png" )


### ####################################
    def write_file_all(self, root_path, frame_id, full_data):
        self.write_file(root_path, frame_id)
        filename = root_path + "particles_frame_all_" + str(frame_id) + ".txt" 
        c = np.savetxt(filename, full_data.ravel(), delimiter ='\n')  
        
    def draw_fig_2d_all(self, root_path, frame_id, xy_max, full_data, force = None, circle = -1):
        self.draw_fig_2d( root_path, frame_id, xy_max, force)
        
        fig, ax = plt.subplots(figsize=(10,10))

        if (circle > 0):
            circle1 = plt.Circle((0, 0), circle, color='r', fill=False)
            ax.add_artist(circle1)

        if (np.array(force).any()):
            X = []; Y = []; U = []; V = []
            for i in range(self.pos.shape[0]):
                if (np.linalg.norm(force[i])>0):
                    X.append(self.before_proj_pos[i][0])
                    Y.append(self.before_proj_pos[i][1])
                    U.append(force[i][0]) 
                    V.append(force[i][1]) 
            ax.quiver(X, Y, U, V, angles='xy', scale_units='xy')
      
        ax.scatter(full_data[:,0], full_data[:,1],  c = 'r', label = "All particles") 
        ax.scatter(self.before_proj_pos[:,0], self.before_proj_pos[:,1], c = 'y', label = "before projection")
        ax.scatter(self.pos[:,0], self.pos[:,1],  c = 'b', label = "after projection")       
        ax.grid()
        fig.legend()
        ax.set_xlim(-xy_max, xy_max)
        ax.set_ylim(-xy_max, xy_max)
    
        if not path.exists(root_path):
            mkdir(root_path) 
        plt.savefig(root_path + "figure_frame_all_" + str(frame_id) + ".png" )



def create_gif(root_path, prefix, frame, name, fps = 10):
	files = [root_path + prefix + str(i) + '.png' for i in range(frame+1) ];
	gif_file = root_path + name +'.gif'
	images = [Image.open(i) for i in files]
	images[0].save(gif_file, save_all=True, append_images=images[1:], optimize=False, duration=1000/fps)
	print ("Created gif in " + gif_file)
