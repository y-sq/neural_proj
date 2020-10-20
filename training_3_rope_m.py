from _dataloader import *
from _training import *
from _constraint_net import *
from _iterative_proj import *

import numpy as np
from mpl_toolkits.mplot3d import Axes3D


NAME = "rope_m"
NUM_PARTICLES = 8
DIMENSION = 2
NUM_ITER = 10
C_LAYERS = [256, 256, 256, 256, 1]
TRAINING_ROOT = "models/tmp/"
TEST_MODEL_ROOT = "models/tmp/"
RESULT_ROOT = "results/"
DATASET_ROOT = "data/rope_d2/"  # A less soft rope 
def DATASET_DATA(i): return DATASET_ROOT + 'data_rope_' + str(i) + '_ori.txt'
def DATASET_LABEL(i): return DATASET_ROOT + 'data_rope_' + str(i) + '.txt'
DATASET_TS = 20
DATASET_SAMPLE_NUM = 7200


def training_main():
    train_opts = {
        "num_epochs": 1000,
        "lr": 1e-3,
        'lr_step': 20,
        'lr_gamma': 0.8,
        "batch_size": 512,
        "loss": 'l1', 
        "weight_decay": 0 
    }

    train_ds, val_ds = get_data_loader(start=0, end=DATASET_SAMPLE_NUM, data_path=DATASET_DATA, label_path=DATASET_LABEL, 
                                       TS=DATASET_TS, P_N=NUM_PARTICLES, DIM=DIMENSION, split=0.9, error_thre=2, thre_type="l1")

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=1).cuda()

    exp_dir = TRAINING_ROOT + NAME + "/"
    print(proj_model)
    train(proj_model, train_ds, val_ds, train_opts=train_opts, exp_dir=exp_dir)


def test_main():
    model_path = TEST_MODEL_ROOT + NAME + "/" + "best_model.pt"

    c_net = MLP_Constraint(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    c_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, constrains=c_net, num_iter=NUM_ITER, stiffness=1)
    # using stiffness = 1
    
    start = DATASET_SAMPLE_NUM-100
    d, l = read_data(start=start, end=DATASET_SAMPLE_NUM, data_path=DATASET_DATA, label_path=DATASET_LABEL, 
                            TS=DATASET_TS, P_N=NUM_PARTICLES, DIM=DIMENSION)

    data = d[:, :, :]
    label = l[:, :, :]
    pred = proj_model(data)
    pred2 = proj_model(pred)
    print(d.size(), l.size(), pred.size())
    pred = pred.detach().numpy()
    pred2 = pred2.detach().numpy()
    data = data.detach().numpy()
    label = label.numpy()

    print(label[0, :, :])

    for i in range(pred.shape[0]):
        plt.scatter(data[i, :, 0], data[i, :, 1], c='y', label = 'before proj')
        plt.scatter(label[i, :, 0], label[i, :, 1], c='g', label = 'gt')
        plt.scatter(pred[i, :, 0], pred[i, :, 1], c='b', label = 'after proj')
        # xy_max = 1.5
        # plt.xlim(-xy_max, xy_max)
        # plt.ylim(-xy_max, xy_max)
        error = np.sum(np.sum((label[i, :, 0] - pred[i, :, 0])**2))
        plt.title("frame: " + str(start) + ' + ' +
                  str(i) + '; error: ' + str(error))
        plt.legend()
        plt.show()
        if (error > 1e-3):
            bad_results_path = RESULT_ROOT + NAME + "_test_bad_results/"
            if not path.exists(bad_results_path):
                mkdir(bad_results_path)
            plt.savefig(bad_results_path + str(start) + '-' + str(i) + '-' + str(error) + '.jpg')
        plt.clf()


if __name__ == '__main__':
    training_main()
    # test_main()
