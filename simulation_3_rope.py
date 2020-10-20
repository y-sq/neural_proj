from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *


MODEL_NAME = "rope"
NUM_PARTICLES = 8
DIMENSION = 2
NUM_ITER = 10
C_LAYERS = [256, 256, 256, 256, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 50
DT = 0.1


def create_simulation():
    NAME = "rope"
    root_path = RESULT_ROOT + NAME + "/"
    
    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=0.9)
    
    pos = np.zeros([NUM_PARTICLES,DIMENSION])
    dx = 0.1
    for i in range(pos.shape[0]):
        pos[i, 0] = i * dx
        pos[i, 1] = 0
    vel = np.array(pos)*0        
    force0 = np.array([-2.0, 8.0]) 
    force7 = np.array([2.0, 8.0])         
    force_g = force_g = np.tile(np.array([0, -2.0]), [NUM_PARTICLES,1])
    force_g[0,:] += force0
    force_g[7,:] += force7
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 2
    simulator.write_file(root_path, 0)
    simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force = force_g
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)
    
    create_gif(root_path, 'figure_frame_', TS, "_" + NAME, 10)

def create_simulation_long():
    NAME = "rope_long"
    root_path = RESULT_ROOT + NAME + "/"

    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    single_proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=0.9)
    groups = [[i*6+k for k in range(8)] for i in range(3)]
    proj_model = GroupProjection(num_particles = 20, dimension = DIMENSION, projs = [single_proj_model], groups = [groups], num_iter=20)
    
    pos = np.zeros([20,DIMENSION])
    dx = 0.1
    for i in range(pos.shape[0]):
        pos[i, 0] = i * dx
        pos[i, 1] = 0
    vel = np.array(pos)*0        
    force0 = np.array([-5.0, 20.0]) 
    force7 = np.array([5.0, 20.0])         
    force_g = force_g = np.tile(np.array([0, -2.0]), [20,1])
    force_g[0,:] += force0
    force_g[19,:] += force7
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 2
    simulator.write_file(root_path, 0)
    simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force = force_g
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)
    
    create_gif(root_path, 'figure_frame_', TS, "_" + NAME, 10)
        
if __name__ == '__main__':
    create_simulation()
    create_simulation_long()