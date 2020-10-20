from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *


MODEL_NAME = "rigid_16"
NUM_PARTICLES = 16
DIMENSION = 2
NUM_ITER = 8
C_LAYERS = [256, 256, 256, 256, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 100
DT = 0.1

root_path = RESULT_ROOT + MODEL_NAME + "/"

def create_simulation():
    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER)
    
    force1 = lambda t: np.array([np.sin(t*2), np.cos(t*2)]) * 15
    force2 = lambda t: np.array([-np.sin(t*2), -np.cos(t*2)]) * 15
    
    pos = np.array([[ -0.0705747,  -0.0477225],
        [ -0.143344, 0.308463],
        [-0.037158,  0.483448],
        [ -0.112173, 0.829575],
        [0.233954,  0.0116891],
        [ 0.166068, 0.207133],
        [0.111511,  0.50575],
        [ 0.160129, 0.709886],
        [0.364264,  -0.065618],
        [ 0.338761, 0.206584],
        [0.360962,  0.466334],
        [0.370712,  0.765977],
        [ 0.670355, 0.0703561],
        [ 0.724735, 0.256815],
        [0.661194,  0.515199],
        [0.669577, 0.747849]])
    vel = np.array(pos)*0
    force = np.array(pos)*0
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 2
    simulator.write_file(root_path, 0)
    simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force[0,:] = force1(ite * DT)
        force[15,:] = force2(ite * DT)
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)

        
if __name__ == '__main__':
    create_simulation()
    create_gif(root_path, 'figure_frame_', TS, "_"+MODEL_NAME, 10)