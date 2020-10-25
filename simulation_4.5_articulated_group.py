from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *
from _evaluate_constraint import *
from numpy.random import rand


ROPE_MODEL_NAME = "rope"
ROPE_NUM_PARTICLES = 8
ROPE_NUM_ITER = 10

RIGID_MODEL_NAME = "rigid_4"
RIGID_NUM_PARTICLES = 4
RIGID_NUM_ITER = 5

C_LAYERS = [256, 256, 256, 256, 1]
DIMENSION = 2
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 50
DT = 0.1

rope_model_path = TEST_MODEL_ROOT + ROPE_MODEL_NAME + "/" + "best_model.pt"
rope_c_net = MLP_Constraint(num_particles=ROPE_NUM_PARTICLES,
                        dimension=DIMENSION, num_features=C_LAYERS)
rope_c_net.load_state_dict(torch.load(rope_model_path))
rope_proj = Projection(num_particles=ROPE_NUM_PARTICLES,
                        dimension=DIMENSION, constrains=rope_c_net, num_iter=ROPE_NUM_ITER, stiffness=0.9)

rigid_model_path = TEST_MODEL_ROOT + RIGID_MODEL_NAME + "/" + "best_model.pt"
rigid_c_net = MLP_Constraint(num_particles=RIGID_NUM_PARTICLES,
                        dimension=DIMENSION, num_features=C_LAYERS)
rigid_c_net.load_state_dict(torch.load(rigid_model_path))
rigid_proj = Projection(num_particles=RIGID_NUM_PARTICLES,
                        dimension=DIMENSION, constrains=rigid_c_net, num_iter=RIGID_NUM_ITER, stiffness=1)

TOTAL_PARTICLES = 28
groups1 = [[i*10+k for k in range(8)] for i in range(3)]
groups2 = [[7+i*10+k for k in range(4)] for i in range(2)]
proj_model = GroupProjection(num_particles=TOTAL_PARTICLES, dimension=DIMENSION,  
                        projs=[rope_proj, rigid_proj], groups = [groups1, groups2], num_iter=10) 

rigid = np.array([[ 0.2514-1,  0.1707],
    [ 0.0483-1, -0.4097],
    [-0.3870-1,  0.0387],
    [-0.4260-1, -0.4451]]) - np.array([ 0.2514-1,  0.1707])

def create_simulation_random_force(idx):
    pos_list = []

    NAME = "articulated_group" + str(idx)
    root_path = RESULT_ROOT + NAME + "/"

    pos = np.zeros([TOTAL_PARTICLES,2])
    dx = 0.1
    rope = np.zeros([8,2])
    for i in range(8):
        rope[i, 0] = - i * dx
        rope[i, 1] = 0
    
    pos[0:8,:] = rope + np.array([2, 0])
    pos[7:11,:] = rigid + pos[7,:]
    pos[10:18,:] = rope + pos[10,:]
    pos[17:21,:] = rigid + pos[17,:]
    pos[20:28,:] = rope + pos[20,:]

    vel = np.array(pos)*0

    force0 = np.array([10., 28.]) + (rand(DIMENSION)-0.5)*4
    force1 = np.array([-10., 28.]) + (rand(DIMENSION)-0.5)*4  

    force = np.tile(np.array([0, -2.]), [TOTAL_PARTICLES,1])
    force[0,:] += force0
    force[-1,:] += force1


    simulator = PBD_Simulation(pos, vel, proj_model)
    
    # xy_max = 3
    # simulator.write_file(root_path, 0)
    # simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        simulator.advect(force, DT)
        # simulator.write_file(root_path, ite+1)
        # simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)
        pos_list.append(simulator.pos)
    
    # create_gif(root_path, 'figure_frame_', TS, "_"+NAME, 10)

    return pos_list


eva_r = Rigid_Body_Eval(rigid)
eva_l = Rope_Length_Eval(0.1)
eva_a = Rope_Angle_Eval(np.pi)

if __name__ == '__main__':
    NUM_SAMPLE = 200
    pos_list = []
    for i in range(NUM_SAMPLE):
        pos_list += create_simulation_random_force(i)
        print(i)
    c_sum_r = 0
    c_sum_l = 0
    c_sum_a = 0
    rope_sum = 0
    rigid_sum = 0
    for pos in pos_list:
        for g in groups1:
            rope = pos[g,:]
            c_sum_l += eva_l.eval(rope)
            c_sum_a += eva_a.eval(rope)
            rope_sum += 1
        for g in groups2:
            ri = pos[g, :]
            c_sum_r += eva_r.eval(ri)
            rigid_sum += 1
    print(c_sum_r / rigid_sum)
    print(c_sum_l / rope_sum)
    print(c_sum_a / rope_sum)
