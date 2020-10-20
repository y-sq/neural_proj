from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *

MODEL_NAME = "rigid_4"
NUM_PARTICLES = 4
DIMENSION = 2
NUM_ITER = 5
C_LAYERS = [256, 256, 256, 256, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 10
DT = 0.1

class Proj_One_Iter():
    def __init__(self, if_cuda = False):
        self.if_cuda = if_cuda
        model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"
        self.c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                                    dimension=DIMENSION, num_features=C_LAYERS)
        self.c_net.load_state_dict(torch.load(model_path))
        if (if_cuda): self.c_net = self.c_net.cuda()
        self.proj_model = Projection(num_particles=NUM_PARTICLES,
                                     dimension=DIMENSION, constrains=self.c_net, num_iter=1)
        if (if_cuda): self.proj_model = self.proj_model.cuda()
    
    def project(self, data):
        d = torch.Tensor(data[None, :, :]).cuda() if self.if_cuda else torch.Tensor(data[None, :, :])
        pred = self.proj_model(d)[0,:,:].detach().cpu().numpy()
        return pred

    def get_c(self, data):
        d = torch.Tensor(data[None, :, :]).cuda() if self.if_cuda else torch.Tensor(data[None, :, :])
        c = self.c_net(d).detach().cpu().numpy()
        return c[0,0]

class Plot_Helper():
    def __init__(self):
        self.c_value = []
        self.delta_x = []
        self.ls_list = ['-', '--', '-.', ':']
    def add_c(self, ite, c):
        if (ite+1 >= len(self.c_value)):
            self.c_value.append([])
        self.c_value[-1].append(c)
    def add_x(self, ite, prev_x, new_x):
        delta_x = np.sum(np.sum((prev_x - new_x)**2))
        if (ite+1 >= len(self.delta_x)):
            self.delta_x.append([])
        self.delta_x[-1].append(delta_x)
    def plot_c(self):
        print(len(self.c_value))
        for i in range(len(self.c_value)):
            c = self.c_value[i]
            x = [k for k in range(len(c))]
            plt.plot(x, c, ls=self.ls_list[i%len(self.ls_list)], label = 'frame ' + str(i+1))
        plt.legend()
        plt.show()
    def plot_delta_x(self):
        for i in range(len(self.delta_x)):
            c = self.delta_x[i]
            x = [k+1 for k in range(len(c))]
            plt.plot(x, c, ls=self.ls_list[i%len(self.ls_list)], label = 'frame ' + str(i+1))
        plt.legend()
        plt.show()

def test_iter():
    pos = np.array([[ 0.2514,  0.1707],
        [ 0.0483, -0.4097],
        [-0.3870,  0.0387],
        [-0.4260, -0.4451]])
    vel = np.array(pos)*0
    force = np.array(pos)*0
    
    force1 = lambda t: np.array([np.sin(t*2), np.cos(t*2)]) * 2.5
    force2 = lambda t: np.array([-np.sin(t*2), -np.cos(t*2)]) * 2.5

    proj = Proj_One_Iter()
    helper = Plot_Helper()

    for ite in range(TS):
        force[0,:] = force1(ite * DT)
        force[3,:] = force2(ite * DT)

        vel = vel + force * DT
        before_proj_pos = pos + vel * DT

        pred = before_proj_pos
        c0 = proj.get_c(pred)
        helper.add_c(ite, c0)
        for i in range(20):
            prev_x = pred
            pred = proj.project(pred)
            c = proj.get_c(pred)
            helper.add_c(ite, c)
            helper.add_x(ite, prev_x, pred)

        vel = (pred - pos) / DT
        pos = pred     
        # plt.scatter(pos[:,0], pos[:,1]); plt.xlim(-1,1); plt.ylim(-1, 1); plt.show()
    
    helper.plot_c()
    helper.plot_delta_x()


class Plot_Shape_Helper():
    def __init__(self):
        self.shapes = []
        self.c_values = []
        self.ls_list = ['-', '--', '-.', ':']
    def add_shape(self, x, c):
        shape = np.zeros([x.shape[0]+1, x.shape[1]])
        shape[2,:] = x[3,:]
        shape[3,:] = x[2,:]
        shape[1,:] = x[1,:]
        shape[0,:] = x[0,:]
        shape[4,:] = x[0,:]
        self.shapes.append(shape)
        self.c_values.append(c)
    def plot_shapes(self, col = 4):
        for i in range(col):
            x = self.shapes[i][:,0] + i
            y = self.shapes[i][:,1]
            plt.plot(x, y, ls=self.ls_list[i%len(self.ls_list)], label = 'c_net(x) = ' + str(self.c_values[i]))
        for i in range(col, len(self.shapes)):
            tmp_col = col // 2
            x = self.shapes[i][:,0] + i%tmp_col
            y = self.shapes[i][:,1] + (i//tmp_col - 1)
            plt.plot(x, y, ls=self.ls_list[i%len(self.ls_list)], label = 'c_net(x) = ' + str(self.c_values[i]))

        plt.xlim(-0.6, col-0.6)
        plt.ylim(-0.6, col-1.6)
        plt.legend()
        plt.show()

def test_iter_shape():    
    helper = Plot_Shape_Helper()
    proj = Proj_One_Iter()    
    shape0 = np.array([[ 0.2514,  0.1707],
        [ 0.0483, -0.4097],
        [-0.3870,  0.0387],
        [-0.4260, -0.4451]])
    for i in range(8):
        shape = shape0 + (np.random.rand(shape0.shape[0], shape0.shape[1]) - 0.5) * i/10 
        c = proj.get_c(shape)
        helper.add_shape(shape, c)
    helper.plot_shapes()
        

if __name__ == '__main__':
    test_iter()   
    test_iter_shape()
