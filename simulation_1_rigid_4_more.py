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
TS = 50
DT = 0.1


def create_simulation_soft():
    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=0.5)
    
    NAME = "rigid_4_soft"
    root_path = RESULT_ROOT + NAME + "/"
    
    force1 = lambda t: np.array([np.sin(t*2), np.cos(t*2)]) * 2.5
    force2 = lambda t: np.array([-np.sin(t*2), -np.cos(t*2)]) * 2.5
    
    pos = np.array([[ 0.2514,  0.1707],
        [ 0.0483, -0.4097],
        [-0.3870,  0.0387],
        [-0.4260, -0.4451]])
    vel = np.array(pos)*0
    force = np.array(pos)*0

    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 1
    simulator.write_file(root_path, 0)
    simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force[0,:] = force1(ite * DT)
        force[3,:] = force2(ite * DT)
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)
    
    create_gif(root_path, 'figure_frame_', TS, "_"+NAME, 10)


def create_simulation_fixed_point():
    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=1, boundary_nodes=[0])
    
    NAME = "rigid_4_fixed0"
    root_path = RESULT_ROOT + NAME + "/"
    
    force1 = lambda t: np.array([-np.sin(t*2), -np.cos(t*2)]) * 2.5
    
    pos = np.array([[ 0.2514,  0.1707],
        [ 0.0483, -0.4097],
        [-0.3870,  0.0387],
        [-0.4260, -0.4451]])
    vel = np.array(pos)*0
    force = np.array(pos)*0

    simulator = PBD_Simulation(pos, vel, proj_model)

    xy_max = 1
    simulator.write_file(root_path, 0)
    simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force[3,:] = force1(ite * DT)
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)
    
    create_gif(root_path, 'figure_frame_', TS, "_"+NAME, 10)


def create_simulation_groups():
    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))

    single_proj = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=1)

    TOTAL_PARTICLES = 8
    proj_model = GroupProjection(num_particles=TOTAL_PARTICLES, dimension=DIMENSION,  # 
                            projs=[single_proj], groups = [[[0,1,2,3],[1,0,7,6],[4,5,6,7]]], num_iter=10) 
    
    NAME = "rigid_4_groups"
    root_path = RESULT_ROOT + NAME + "/"

    pos = np.zeros([8,2])
    pos[0:4,:] = np.array([[ 0.2514-0.5,  0.1707],
        [ 0.0483-0.5, -0.4097],
        [-0.3870-0.5,  0.0387],
        [-0.4260-0.5, -0.4451]])
    pos[4:8,:] = (pos[0:4,:] + np.array([1.1, 0]))
    tmp = torch.Tensor(pos[None, :, :])
    pos = proj_model(tmp)[0,:,:].detach().numpy()

    vel = np.array(pos)*0
    force = np.array(pos)*0

    simulator = PBD_Simulation(pos, vel, proj_model)
        
    force1 = lambda t: np.array([np.sin(t*2), np.cos(t*2)]) * 5
    force2 = lambda t: np.array([-np.sin(t*2), -np.cos(t*2)]) * 5
    
    xy_max = 2
    simulator.write_file(root_path, 0)
    simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force[4,:] = force1(ite * DT)
        force[3,:] = force2(ite * DT)
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)
    
    create_gif(root_path, 'figure_frame_', TS, "_"+NAME, 10)


def create_simulation_groups_2():
    model_path = TEST_MODEL_ROOT + MODEL_NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))

    single_proj = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=1)
    single_proj_fixed = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=1, boundary_nodes=[3])

    TOTAL_PARTICLES = 8
    proj_model = GroupProjection(num_particles=TOTAL_PARTICLES, dimension=DIMENSION,  # 
                            projs=[single_proj_fixed, single_proj], groups = [[[0,1,2,3]],[[1,0,7,6],[4,5,6,7]]], num_iter=10) 
    
    NAME = "rigid_4_groups2"
    root_path = RESULT_ROOT + NAME + "/"

    pos = np.zeros([8,2])
    pos[0:4,:] = np.array([[ 0.2514,  0.1707],
        [ 0.0483, -0.4097],
        [-0.3870,  0.0387],
        [-0.4260, -0.4451]])
    pos[4:8,:] = (pos[0:4,:] + np.array([1.1, 0]))
    tmp = torch.Tensor(pos[None, :, :])
    pos = proj_model(tmp)[0,:,:].detach().numpy()

    vel = np.array(pos)*0
    force = np.array(pos)*0

    simulator = PBD_Simulation(pos, vel, proj_model)
        
    force1 = lambda t: np.array([np.sin(t), np.cos(t)]) * 5
    
    xy_max = 3
    simulator.write_file(root_path, 0)
    simulator.draw_fig_2d(root_path, 0, xy_max, force = None)
    for ite in range(TS):
        force[4,:] = force1(ite * DT)
        simulator.advect(force, DT)
        simulator.write_file(root_path, ite+1)
        simulator.draw_fig_2d(root_path, ite+1, xy_max, force = force)
    
    create_gif(root_path, 'figure_frame_', TS, "_"+NAME, 10)

        
if __name__ == '__main__':
    create_simulation_fixed_point()
    create_simulation_soft()
    create_simulation_groups()
    create_simulation_groups_2()