from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *
from _evaluate_constraint import *
from numpy.random import rand

MODEL_NAME = "rigid_4"
NUM_PARTICLES = 4
DIMENSION = 2
NUM_ITER = 5
C_LAYERS = [256, 256, 256, 256, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 50
DT = 0.1


model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                        dimension=DIMENSION, num_features=C_LAYERS)
c_net.load_state_dict(torch.load(model_path))
proj_model = Projection(num_particles=NUM_PARTICLES,
                        dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER)

init_shape = np.array([[ 0.2514,  0.1707],
        [ 0.0483, -0.4097],
        [-0.3870,  0.0387],
        [-0.4260, -0.4451]])

def create_simulation_random_two_force(idx):
    pos_list = []
    
    NAME = MODEL_NAME + str(idx)
    root_path = RESULT_ROOT + NAME + "/"
    
    force1 = lambda t: np.array([np.sin(t*2), np.cos(t*2)]) * 2.5 + (rand(DIMENSION)-0.5)
    force2 = lambda t: np.array([-np.sin(t*2), -np.cos(t*2)]) * 2.5 + (rand(DIMENSION)-0.5)  
    
    pos = np.array(init_shape)
    vel = np.array(pos)*0
    force = np.array(pos)*0
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    # xy_max = 1
    # simulator.write_file(root_path, 0)
    # simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force[0,:] = force1(ite * DT)
        force[3,:] = force2(ite * DT)
        simulator.advect(force, DT)
        # simulator.write_file(root_path, ite+1)
        # simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)

        pos_list.append(simulator.pos)    
    
    return pos_list
    # create_gif(root_path, 'figure_frame_', TS, "_"+MODEL_NAME, 10)


eva = Rigid_Body_Eval(init_shape)

if __name__ == '__main__':
    NUM_SAMPLE = 200
    pos_list = []
    for i in range(NUM_SAMPLE):
        pos_list += create_simulation_random_two_force(i)
        print(i)
    c_sum = 0
    for pos in pos_list:
        c_sum += eva.eval(pos)
    print(c_sum / len(pos_list))
