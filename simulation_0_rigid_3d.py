from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *


MODEL_NAME = "rigid_3d"
NUM_PARTICLES = 8
DIMENSION = 3
NUM_ITER = 8
C_LAYERS = [256, 256, 256, 256, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 100
DT = 0.1

NAME = "rigid_3d"

root_path = RESULT_ROOT + MODEL_NAME + "/"

def create_simulation():
    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER)
    
    force1 = lambda t: np.array([np.sin(t), np.cos(t), np.cos(t)]) * 2.5
    force2 = lambda t: np.array([-np.sin(t), -np.cos(t), -np.cos(t)]) * 2.5
    
    pos = np.zeros([NUM_PARTICLES, DIMENSION])
    for i in range(2): 
        for j in range(2): 
            for k in range(2): 
                pos[i*4+j*2+k][0] = i*0.5 - 0.25; pos[i*4+j*2+k][1] = j*0.5 - 0.25; pos[i*4+j*2+k][2] = k*0.5 - 0.25; 

    vel = np.array(pos)*0
    force = np.array(pos)*0
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 1
    simulator.write_file(root_path, 0)
    simulator.draw_fig_3d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force[0,:] = force1(ite * DT)
        force[7,:] = force2(ite * DT)
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_3d(root_path, ite+1, xy_max, force = force)

        
if __name__ == '__main__':
    create_simulation()
    create_gif(root_path, 'figure_frame_', TS, "_"+NAME, 10)