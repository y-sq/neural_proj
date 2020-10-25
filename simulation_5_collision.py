from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *
from _evaluate_constraint import *
from numpy.random import rand


MODEL_NAME = "collision"
NUM_PARTICLES = 8
DIMENSION = 2
NUM_ITER = 10
C_LAYERS = [512, 512, 512, 512, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 50
DT = 0.1

rigid = np.array([[ 0.130514,   0.437662 ],
                    [ 0.0862001,  0.882675 ],
                    [ 0.575526,   0.481976 ],
                    [ 0.531213,   0.926988 ]])  

model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                        dimension=DIMENSION, num_features=C_LAYERS)
c_net.load_state_dict(torch.load(model_path))
proj_model = Projection(num_particles=NUM_PARTICLES,
                        dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER)

R = 2
r = 0.1

def create_simulation_random_vel(idx):
    pos_list = []

    NAME = "collision"+str(idx)
    root_path = RESULT_ROOT + NAME + "/"
    
    pos = np.array([[ 0.130514,   0.437662 ],
                    [ 0.0862001,  0.882675 ],
                    [ 0.575526,   0.481976 ],
                    [ 0.531213,   0.926988 ],
                    [-0.457831,  -0.12947  ],
                    [-0.5583,     0.306311 ],
                    [-0.0220487, -0.0290008],
                    [-0.122518,   0.406781 ]])
    vel = np.array([[5.0, 0.0],[5.0, 0.0],[5.0, 0.0],[5.0, 0.0],
                    [5.0, 0.0],[5.0, 0.0],[5.0, 0.0],[5.0, 0.0]]) * 0.4
    for i in range(2):
        vel[i*4: (i+1)*4] += np.tile((rand(DIMENSION)-0.5), [4,1])

    # print(vel)

    force_g = np.tile(np.array([0, -5]), [8,1])
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    # xy_max = 2
    # data = simulator.pos
    # full_data = np.zeros([32,2])
    # full_data[0:16, :] = get_full_data(get_boundary(data[0:4,:]))
    # full_data[16:32, :] = get_full_data(get_boundary(data[4:8,:]))
    # simulator.write_file_all(root_path, 0, full_data)
    # simulator.draw_fig_2d_all(root_path, 0, xy_max, full_data, force = None, circle = 2)
    
    for ite in range(TS):
        force = force_g
        simulator.advect(force, DT)
        pos_list.append(simulator.pos)
    #     data = simulator.pos
    #     full_data = np.zeros([32,2])
    #     full_data[0:16, :] = get_full_data(get_boundary(data[0:4,:]))
    #     full_data[16:32, :] = get_full_data(get_boundary(data[4:8,:]))    
    #     simulator.write_file_all(root_path, ite+1, full_data)
    #     simulator.draw_fig_2d_all(root_path, ite+1, xy_max, full_data, force = force, circle = 2)

    # create_gif(root_path, 'figure_frame_', TS, "_" + MODEL_NAME, 10)
    # create_gif(root_path, 'figure_frame_all_', TS, "_" + MODEL_NAME, 10)
    return pos_list

# used for visualization
def get_boundary(new_data):
    center1 = (new_data[0,:]+new_data[1,:]+new_data[2,:]+new_data[3,:])/4
    boundary_data = np.zeros([4,2])
    boundary_data[0, :] = 0.6 * (new_data[3,:]-center1) - 1.2 * (new_data[1,:]-center1) + center1
    boundary_data[1, :] = 1.2 * (new_data[3,:]-center1) + 0.6 * (new_data[1,:]-center1) + center1
    boundary_data[2, :] = - 0.6 * (new_data[3,:]-center1) + 1.2 * (new_data[1,:]-center1) + center1
    boundary_data[3, :] = - 1.2 * (new_data[3,:]-center1) - 0.6 * (new_data[1,:]-center1) + center1
    return boundary_data

def get_full_data(boundary_data):
    full_data = np.zeros([16,2])
    x1 = boundary_data[3,0]
    y1 = boundary_data[3,1]
    x2 = boundary_data[1,0]
    y2 = boundary_data[1,1]
    for i in range(4):
        for j in range(4):
            temp1 = (j * boundary_data[3,:] + (3-j) * boundary_data[2,:] ) / 3
            temp2 = (j * boundary_data[0,:] + (3-j) * boundary_data[1,:] ) / 3
            full_data[i*4+j, :] = (i * temp1 + (3-i) * temp2) / 3
    return full_data


rigid_eval = Rigid_Body_Eval(rigid)
boundary_coli = Collision_Boundary_Eval(R-r)
object_coli = Collision_Particle_Eval(r)

# tmp = get_boundary(rigid)
# plt.scatter(tmp[:,0], tmp[:,1])
# plt.show()

if __name__ == '__main__':
    NUM_SAMPLE = 200
    pos_list = []
    for i in range(NUM_SAMPLE):
        pos_list += create_simulation_random_vel(i)
        print(i)

    c_shape = 0
    c_collision = 0
    shape_sum = 0
    collision_sum = 0

    TOTAL = 2

    for pos in pos_list:
        for i in range(TOTAL):
            c_shape += rigid_eval.eval(pos[i*4: (i+1)*4, :])
            bou = get_boundary(pos[i*4: (i+1)*4, :])
            c_collision += boundary_coli.eval(bou)
            shape_sum += 1
            collision_sum += 1
        for i in range(TOTAL):
            for j in range(i+1, TOTAL):
                bou1 = get_boundary(pos[i*4: (i+1)*4, :])
                bou2 = get_boundary(pos[j*4: (j+1)*4, :])
                c_collision += object_coli.eval(bou1, bou2)
                collision_sum += 1

    print(c_shape / shape_sum)
    print(c_collision / collision_sum)