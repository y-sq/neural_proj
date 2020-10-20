from _constraint_net import *
from _iterative_proj import *
from _run_simulation import *


MODEL_NAME = "collision"
NUM_PARTICLES = 8
DIMENSION = 2
NUM_ITER = 10
C_LAYERS = [512, 512, 512, 512, 1]
TEST_MODEL_ROOT = "models/"
RESULT_ROOT = "results/"
TS = 50
DT = 0.1


def create_simulation():
    NAME = "collision"
    root_path = RESULT_ROOT + NAME + "/"

    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER)
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
    force_g = np.tile(np.array([0, -5]), [8,1])
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 2
    
    data = simulator.pos
    full_data = np.zeros([32,2])
    full_data[0:16, :] = get_full_data(get_boundary(data[0:4,:]))
    full_data[16:32, :] = get_full_data(get_boundary(data[4:8,:]))
    simulator.write_file_all(root_path, 0, full_data)
    simulator.draw_fig_2d_all(root_path, 0, xy_max, full_data, force = None, circle = 2)
    
    for ite in range(TS):
        force = force_g
        simulator.advect(force, DT)
        
        data = simulator.pos
        full_data = np.zeros([32,2])
        full_data[0:16, :] = get_full_data(get_boundary(data[0:4,:]))
        full_data[16:32, :] = get_full_data(get_boundary(data[4:8,:]))    
        simulator.write_file_all(root_path, ite+1, full_data)
        simulator.draw_fig_2d_all(root_path, ite+1, xy_max, full_data, force = force, circle = 2)

    # create_gif(root_path, 'figure_frame_', TS, "_" + MODEL_NAME, 10)
    create_gif(root_path, 'figure_frame_all_', TS, "_" + MODEL_NAME, 10)


def create_simulation_more():
    NAME = "collision_more"
    root_path = RESULT_ROOT + NAME + "/"

    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    single_proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER)
    groups = []
    for i in range(4):
        for j in range(i+1,4):
            groups.append([i*4+k for k in range(4)] + [j*4+k for k in range(4)])
    proj_model = GroupProjection2(16, DIMENSION, [single_proj_model], [groups], 10)

    rigid = np.array([[ 0.130514,   0.437662 ],
                        [ 0.0862001,  0.882675 ],
                        [ 0.575526,   0.481976 ],
                        [ 0.531213,   0.926988 ]])    
    pos = np.zeros([16,2])
    pos[0:4, :] = rigid + np.array([-1.2,-0.5])
    pos[4:8, :] = rigid + np.array([0.25,-0.5])
    pos[8:12, :] = rigid + np.array([-0.2,0.6])
    pos[12:16, :] = rigid + np.array([-1.3,0.5])
    vel = np.array([[3.0, 0.0],[3.0, 0.0],[3.0, 0.0],[3.0, 0.0],
                    [1.0, 0.0],[1.0, 0.0],[1.0, 0.0],[1.0, 0.0],
                    [4.0, 0.0],[4.0, 0.0],[4.0, 0.0],[4.0, 0.0],
                    [4.0, 0.0],[4.0, 0.0],[4.0, 0.0],[4.0, 0.0]]) * 0.3
    force_g = np.tile(np.array([0, -5]), [16,1])
    
    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 2
    
    data = simulator.pos
    full_data = np.zeros([64,2])
    for i in range(4):
        full_data[i*16:(i+1)*16, :] = get_full_data(get_boundary(data[i*4:(i+1)*4,:]))
    simulator.write_file_all(root_path, 0, full_data)
    simulator.draw_fig_2d_all(root_path, 0, xy_max, full_data, force = None, circle = 2)
    
    for ite in range(TS):
        force = force_g
        simulator.advect(force, DT)
        
        data = simulator.pos
        for i in range(4):
            full_data[i*16:(i+1)*16, :] = get_full_data(get_boundary(data[i*4:(i+1)*4,:]))
        simulator.write_file_all(root_path, ite+1, full_data)
        simulator.draw_fig_2d_all(root_path, ite+1, xy_max, full_data, force = force, circle = 2)

    # create_gif(root_path, 'figure_frame_', TS, "_" + MODEL_NAME, 10)
    create_gif(root_path, 'figure_frame_all_', TS, "_" + MODEL_NAME, 10)


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


if __name__ == '__main__':
    create_simulation()
    # create_simulation_more()