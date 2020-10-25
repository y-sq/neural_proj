from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *
from _evaluate_constraint import *
from numpy.random import rand


MODEL_NAME = "rope"
NUM_PARTICLES = 8
DIMENSION = 2
NUM_ITER = 10
C_LAYERS = [256, 256, 256, 256, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 50
DT = 0.1

model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                        dimension=DIMENSION, num_features=C_LAYERS)
c_net.load_state_dict(torch.load(model_path))
single_proj_model = Projection(num_particles=NUM_PARTICLES,
                        dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=0.9)
groups = [[i*6+k for k in range(8)] for i in range(3)]
proj_model = GroupProjection(num_particles = 20, dimension = DIMENSION, projs = [single_proj_model], groups = [groups], num_iter=20)


def create_simulation_random_two_force(idx):
    pos_list = []

    NAME = "rope_long" + str(idx)
    root_path = RESULT_ROOT + NAME + "/"
    
    pos = np.zeros([20,DIMENSION])
    dx = 0.1
    for i in range(pos.shape[0]):
        pos[i, 0] = i * dx
        pos[i, 1] = 0
    vel = np.array(pos)*0        

    force0 = np.array([-5.0, 20.0]) + (rand(DIMENSION)-0.5) * 2
    force7 = np.array([5.0, 20.0]) + (rand(DIMENSION)-0.5) * 2         
    force_g = force_g = np.tile(np.array([0, -2.0]), [20,1])
    force_g[0,:] += force0
    force_g[19,:] += force7
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    # xy_max = 2
    # simulator.write_file(root_path, 0)
    # simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force = force_g
        simulator.advect(force, DT)
        pos_list.append(simulator.pos)
    #     simulator.write_file(root_path, ite+1)
    #     simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)
    
    # create_gif(root_path, 'figure_frame_', TS, "_" + NAME, 10)
    return pos_list


eva_l = Rope_Length_Eval(0.1)
eva_a = Rope_Angle_Eval(np.pi)

if __name__ == '__main__':
    NUM_SAMPLE = 200
    pos_list = []
    for i in range(NUM_SAMPLE):
        pos_list += create_simulation_random_two_force(i)
        print(i)
    c_sum_l = 0
    c_sum_a = 0
    for pos in pos_list:
        c_sum_l += eva_l.eval(pos)
        c_sum_a += eva_a.eval(pos)
    print(c_sum_l / len(pos_list))
    print(c_sum_a / len(pos_list))