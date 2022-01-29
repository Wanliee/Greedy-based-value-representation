import torch as th
import torch.nn as nn
from torch.optim import RMSprop
import random

# verification of calculated joint Q value function (Eq.3, Section 3.1)
factorization = "mvd"           # mvd or lvd

# agent
class Q_mat(nn.Module):
    def __init__(self, mat_size):
        super(Q_mat, self).__init__()
    
        self.Q_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, mat_size))
    
    def forward(self, inp):
        return self.Q_net(inp)

class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
    
        self.mixer_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
        self.v_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, inp, q_1, q_2):
        return th.abs(self.mixer_net(inp))[0] * q_1 + th.abs(self.mixer_net(inp))[1] * q_2 + self.v_net(inp)
    
# env
matrix = th.tensor([[8,-12,-12],[-12,0,0],[-12,0,6]]).float()
mat_size = 3
episodes = 500
batch_size = 100
lr = 0.001
epsl = 0.2

Q_matrix = th.randint(-10,10,(mat_size, mat_size)).float()
Q_matrix_cal = th.randint(-10,10,(mat_size, mat_size)).float()

q_in = th.ones(4)
agent_1 = Q_mat(mat_size)
agent_2 = Q_mat(mat_size)

params = list(agent_1.parameters())
params += list(agent_2.parameters())
if factorization == "mvd":
    mixer = Mixer()
    params += list(mixer.parameters())

optimiser = RMSprop(params=params, lr=lr)

print_calculation=True
corrupt_after_stable=False
only_run_calculation=False
independent_explore=True
use_loss_weight=True
detach_greedy=False
show_ep_and_w=False
only_cal_w = False
fix_greedy = False


cal_loss = []
for ep in range(episodes):
    loss = 0
    for i in range(batch_size):
        q_out1 = agent_1.forward(q_in)
        q_out2 = agent_2.forward(q_in)

        if independent_explore:
            rand = random.random()
            if rand < epsl:                 # exploration
                ac_1 = th.randint(0,mat_size,()).long()
            else:                           
                ac_1 = q_out1.max(0)[1].long()

            rand = random.random()
            if rand < epsl:        
                ac_2 = th.randint(0,mat_size,()).long()
            else:                           # exploration
                ac_2 = q_out2.max(0)[1].long()

        Gt = matrix[ac_1, ac_2]

        if factorization == "mvd":
            q_tot = mixer(q_in, q_out1[ac_1], q_out2[ac_2])
        elif factorization == "lvd":
            q_tot = q_out1[ac_1] + q_out2[ac_2]

        error = Gt - q_tot
        loss +=  (error) ** 2       
    
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


    print("episode:",ep)
    print("Gt", matrix[q_out1.max(0)[1],q_out2.max(0)[1]].item())

    for x in range(mat_size):
        for y in range(mat_size):
            if factorization == "mvd":
                Q_matrix[x][y] = mixer(q_in, q_out1[x], q_out2[y])
            else:
                Q_matrix[x][y] = q_out1[x] + q_out2[y]
    
    print("ac:", q_out1.max(0)[1].item()+1, q_out2.max(0)[1].item()+1)

    if print_calculation:
        if not fix_greedy:
            i_cal = q_out1.max(0)[1]
            j_cal = q_out2.max(0)[1]
        for x in range(mat_size):
            a = matrix[x, j_cal]
            for y in range(mat_size):
                # print calculation result (Eq.3, Section 3.1)
                Q_matrix_cal[x][y] = epsl/mat_size*(matrix[x,:].sum().item()+matrix[:,y].sum().item())+\
                    (1-epsl)*(matrix[x, j_cal]+matrix[i_cal, y]).item()-epsl**2/mat_size**2*matrix.sum().item()-\
                        (1-epsl)*epsl/mat_size*(matrix[i_cal,:].sum().item()+matrix[:,j_cal].sum().item())-(1-epsl)**2*matrix[i_cal,j_cal].item()
                
        cal_error = Q_matrix.sum().item()-Q_matrix_cal.sum().item()
        cal_error = cal_error ** 2
        cal_loss.append([ep, cal_error])
        ep += 1

        print("cal_error:",cal_error)

        print("Q_mat_cal\n", Q_matrix_cal)
        print("Q_mat\n", Q_matrix)