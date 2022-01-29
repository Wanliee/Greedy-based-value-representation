from os import EX_SOFTWARE
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
import random

# a simple estimation of its and gvr (without trade-off)
# algo = gvr, its or vdn
algo = "gvr"

# generate the payoff matrix
th.manual_seed(1534234326)
matrix = th.randint(-20,7,(3,3,3,3)).float()
matrix[0,0,0,0] = 8.0
matrix[n_actions-1,n_actions-1,n_actions-1,n_actions-1] = 6

# agent
class Q_mat(nn.Module):
    def __init__(self, mat_size):
        super(Q_mat, self).__init__()
    
        self.Q_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, mat_size))
        
    def forward(self, inp):
        return self.Q_net(inp)


class Value_net(nn.Module):
    def __init__(self):
        super(Value_net, self).__init__()
        self.Vnet = nn.Sequential(nn.Linear(4,32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, batch_size):
        batch_state = th.ones(4).repeat(batch_size,1)
        return th.abs(self.Vnet(batch_state))


class Replay_Buffer():
    def __init__(self, batch_size, max_buffer_length):
        self.buffer = {
            "acs": th.tensor([]).long(),
            "Gt": th.tensor([])
        }
        self.max_length = max_buffer_length
        self.batch_size = batch_size

    def add_sample(self, acs, Gt):
        self.buffer["acs"] = th.cat((self.buffer["acs"], acs.unsqueeze(0)), dim=0)
        self.buffer["Gt"] = th.cat((self.buffer["Gt"], Gt.float().unsqueeze(0)), dim=0)
        if self.buffer["acs"].shape[0] > self.max_length:
            self.buffer["acs"] = self.buffer["acs"][1:]
            self.buffer["Gt"] = self.buffer["Gt"][1:]

    def sample(self):
        buffer_length = self.buffer["Gt"].shape[0]
        ep_ids = self.buffer["Gt"].max(0)[1]
        sup_acs = self.buffer["acs"][ep_ids].squeeze(0)
        sup_Gts = self.buffer["Gt"][ep_ids].squeeze(0).clone()
        self.buffer["Gt"][ep_ids] = 0.0
        return sup_acs,sup_Gts

    def can_sample(self):
        buffer_length = self.buffer["Gt"].shape[0]
        return (buffer_length >= self.batch_size)

# env
episodes = 200
batch_size = 100
lr = 0.005

alpha = 0.1

n_actions = 3
n_agents = 4

for ep_times in range(100):
    epsl = ep_times / 100 + 0.9
    
    count_opt = 0
    for _ in range(100):
        agents = []
        if algo == "gvr":
            vnet = Value_net()
            sup_buffer = Replay_Buffer(1, 10)
            sup_optimiser = RMSprop(params=list(vnet.parameters()), lr=0.001)
            
        for i in range(n_agents):
            agents.append(Q_mat(n_actions))
        params = list(agents[0].parameters())
        for i in range(n_agents-1):
            params += list(agents[i+1].parameters())

        q_in = th.ones(4)
        optimiser = RMSprop(params=params, lr=lr)

        for ep in range(episodes):
            loss = 0
            for i in range(batch_size):
                qouts = th.tensor([])
                acs = th.tensor([]).long()
                for a in range(n_agents):
                    qouts = th.cat((qouts, agents[a].forward(q_in).unsqueeze(0)),dim=0)
                    rand = random.random()
                    if rand < epsl:                 # exploration
                        acs = th.cat((acs, th.randint(0,n_actions,()).long().unsqueeze(0)),dim=0) 
                    else:                           # exploitation
                        acs = th.cat((acs, qouts[a].max(0)[1].unsqueeze(0)),dim=0)

                Gt = matrix[acs[0], acs[1], acs[2], acs[3]]
                q_tot = th.gather(qouts, dim=-1, index=acs.unsqueeze(-1)).sum()
                greedy_actions = qouts.max(1)[1]
                greedy = th.where(greedy_actions == acs, th.tensor(1.0), th.tensor(0.0))

                num_greedy = greedy.sum()
                full_greedy = (num_greedy == n_agents)
                joint_q_greedy = qouts.max(1)[0].sum()

                if algo == "gvr":
                    v = vnet.forward(1)
                    if full_greedy:             # train critic
                        for _ in range(3):
                            v = vnet.forward(1)
                            v_error = Gt - v
                            v_loss = v_error ** 2
                            sup_optimiser.zero_grad()
                            v_loss.backward()
                            sup_optimiser.step()
                    elif Gt > v:
                        sup_buffer.add_sample(acs, Gt)               

                if algo=="vdn":
                    error = Gt - q_tot
                    loss += (error/20) ** 2

                elif algo=="its":
                    target = Gt
                    if Gt < joint_q_greedy and (not full_greedy):
                        target = joint_q_greedy - th.abs(joint_q_greedy) * alpha
                    error = target.detach() - q_tot
                    loss += (error/20) ** 2

                if algo == "gvr":
                    # stage 1
                    target = Gt
                    if Gt < joint_q_greedy and (not full_greedy):
                        target = joint_q_greedy - th.abs(joint_q_greedy) * alpha
                    error = target.detach() - q_tot
                    loss += (error/20) ** 2

                    # stage 2
                    if sup_buffer.can_sample():
                        sup_acs, sup_Gt = sup_buffer.sample()

                        v = vnet.forward(1)
                        sup_qtot = th.gather(qouts, dim=-1, index=sup_acs.unsqueeze(-1)).sum()
                        sup_greedy = th.where(greedy_actions == sup_acs, th.tensor(1.0), th.tensor(0.0))
                        sup_num_greedy = sup_greedy.sum()
                        sup_full_greedy = (sup_num_greedy == n_agents)

                        npg = epsl/n_actions
                        pg = 1-epsl+epsl/n_actions
                        eta1 = npg**(n_agents-1)
                        eta2 = pg**(n_agents-1)
                        eta1_1 = pg**(sup_num_greedy-1) * npg**(n_agents-sup_num_greedy) 
                        e_q = (sup_Gt - v) / (v + 1e-7)
            
                        w_ser = th.where(sup_num_greedy == 0, (alpha/e_q)*(eta2-eta1)-eta1, (alpha/e_q)*(eta2-eta1_1)-eta1_1).squeeze()
                        w_ser = th.where(w_ser > 0, w_ser, th.tensor(0.0))

                        sup_target = sup_Gt
                        if sup_Gt < joint_q_greedy and (not sup_full_greedy):
                            sup_target = joint_q_greedy - th.abs(joint_q_greedy) * alpha
                            
                        sup_error = sup_target.detach() - sup_qtot
                        loss += w_ser.detach() * (sup_error/20) ** 2

                        if sup_Gt > joint_q_greedy and (not sup_full_greedy):
                            sup_buffer.add_sample(sup_acs, sup_Gt)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()              

            print("\n episode:",ep,"----------------")
            if algo == "gvr":
                print("\n epsl:", epsl,"\n Q_jt:",joint_q_greedy.item(),"\n value:",v.item(),"\n buffer data:", sup_buffer.buffer["Gt"],"\n acs:",greedy_actions) 
            else:
                print("\n epsl:", epsl,"\n Q_jt:",joint_q_greedy.item(),"\n acs:",greedy_actions) 

        if (qouts[0].max(0)[1] == 0)*(qouts[1].max(0)[1] == 0)*(qouts[2].max(0)[1]==0)*(qouts[3].max(0)[1]==0):
            count_opt += 1

    # with open("./gvr_seed12.txt", mode="a+") as f:
    #     f.write(str([epsl, count_opt])+",")

print("finish!")
