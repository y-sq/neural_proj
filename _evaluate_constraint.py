import numpy as np

class Rigid_Body_Eval():
    def __init__ (self, init_shape):
        self.init_shape = init_shape
    def eval(self, data):
        sum_l2 = 0
        cnt = 0
        for i in range (len(data)):
            for j in range (i+1, len(data)):
                gt = np.linalg.norm(self.init_shape[i,:] - self.init_shape[j,:])
                test = np.linalg.norm(data[i,:] - data[j,:])
                sum_l2 += (gt - test) ** 2 
                cnt += 1
        return sum_l2/cnt

class Rope_Length_Eval():
    def __init__ (self, init_len):
        self.init_len = init_len
    def eval(self, data):
        sum_l2 = 0
        cnt = 0
        for i in range (len(data) - 1):
            gt = self.init_len
            test = np.linalg.norm(data[i,:] - data[i+1,:])
            sum_l2 += (gt - test) ** 2 
            cnt += 1
        return sum_l2/cnt

class Rope_Angle_Eval():
    def __init__ (self, init_ang):
        self.init_ang = init_ang
    def eval(self, data):
        sum_l2 = 0
        cnt = 0
        for i in range (len(data) - 2):
            gt = self.init_ang
            t1 = data[i,:] - data[i+1,:]
            t2 = data[i+2,:] - data[i+1,:]
            cos = np.clip(np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2)), -1, 1)
            test = np.arccos(cos)
            sum_l2 += (gt - test) ** 2 
            cnt += 1
        return sum_l2/cnt

# only used for circle boundary centered at origin
class Collision_Boundary_Eval():
    def __init__ (self, min_dis):
        self.min_dis = min_dis
    def eval (self, data):
        sum_l2 = 0
        cnt = 0
        for i in range (len(data)):
            gt = self.min_dis
            test = np.linalg.norm(data[i,:])
            temp = test - gt
            if temp < 0: # distance to center is smaller than radis
                temp = 0
            sum_l2 += (temp) ** 2 
            cnt += 1
        return sum_l2/cnt


class Collision_Particle_Eval():
    def __init__ (self, r):
        self.min_dis = r*2
    def eval (self, r1, r2):
        sum_l2 = 0
        cnt = 0
        for i in range(r1.shape[0]):
            for j in range(r2.shape[0]):
                gt = self.min_dis
                test = np.linalg.norm(r1[i,:] - r2[j, :])
                temp = test - gt
                if temp > 0: # distance to each other is larger than radius
                    temp = 0
                sum_l2 += (temp) ** 2 
                cnt += 1        
        return sum_l2/cnt




