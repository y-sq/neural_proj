from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *


MODEL_NAME = "articulated"
NUM_PARTICLES = 10
DIMENSION = 2
NUM_ITER = 8
C_LAYERS = [256, 256, 256, 256, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 50
DT = 0.1

root_path = RESULT_ROOT + MODEL_NAME + "/"

def create_simulation():
    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER)
    
    force1 = lambda t: np.array([0.6, 1]) * 10 
    
    data = np.zeros([10,2])
    dx = 0.2
    for i in range(data.shape[0]-3):
        data[i, 0] = i * dx
        data[i, 1] = 0
    data[7,0] = data[6,0] + .35; data[7,1] = data[6,1] + (-.1); 
    data[8,0] = data[6,0] + .55; data[8,1] = data[6,1] + .4; 
    data[9,0] = data[6,0] + .15; data[9,1] = data[6,1] + .45; 
    pos = data
    vel = np.array(pos)*0
    force = np.array(pos)*0
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 2
    simulator.write_file(root_path, 0)
    simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force[8,:] = force1(ite * DT)
        prev_x = simulator.pos
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)

        
if __name__ == '__main__':
    create_simulation()
    create_gif(root_path, 'figure_frame_', TS, "_"+MODEL_NAME, 10)