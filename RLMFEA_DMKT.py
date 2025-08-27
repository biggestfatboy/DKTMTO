import copy
import openpyxl
import numpy as np
import os
import copy as cp
from CEC2017MTSO import *
from evo_operator import *
from MLP_diffusion_model import *
import torch
import random
from torch.utils.data import DataLoader
from Data_construct import MTO_data
from tasks import *
import pandas as pd
import time

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
class RLMFEA:
    def __init__(self,popsize, max_d,taskf1,task1d, taskf2,task2d, maxfes, KT_interval,train_interval,transfer_num, net_info,task_bound):
        self.MAXFES = maxfes
        self.fes1 = 0
        self.fes2 = 0
        self.pop_size = popsize
        self.max_d = max(task1d,task2d)
        self.taskf1 = taskf1
        self.taskf2 = taskf2
        self.task1_d = task1d
        self.task2_d = task2d
        self.task1_num = 0
        self.task2_num = 0
        self.F = 0.5
        self.cr = 0.6
        self.mu = 10
        self.mum = 5
        self.record_allbest_t1 = []
        self.record_allbest_t2 = []
        self.task1_best = float('inf')
        self.task1_best_position = None
        self.task2_best = float('inf')
        self.task2_best_position = None
        self.rmp_choices = [0.1*i for i in range(1,8)]
        self.task1_rmp = 0.3
        self.task1_action=2
        self.task2_action=2
        self.task2_rmp = 0.3
        self.qtable1 = np.zeros((3, 7))
        self.qtable2 = np.zeros((3, 7))
        self.beta = 0.9
        self.alpha = 0.1
        self.train_interval = train_interval
        self.net_info = net_info
        self.transfer_num = transfer_num
        self.KT_interval = [float(KT_interval), float(KT_interval)]
        self.transfer1 = False
        self.transfer2 = False
        self.task_bound1 = task_bound['task1']
        self.task_bound2 = task_bound['task2']
        self.archive_pop1 = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2 = np.zeros((self.pop_size, self.max_d))
        self.gen = 0
    def operator_assign(self):
        operator1 = None
        operator2 = None
        if np.random.rand()<0.5:
            operator1 = 'GA'
            operator2 = 'DE'
        else:
            operator1 = 'DE'
            operator2 = 'GA'

        return operator1, operator2
    def pop_init(self):
        # 初始化种群

        self.task1_pop = []
        self.task2_pop = []
        self.task1_totalfit = 0
        self.task2_totalfit = 0
        # 每个个体初始化有五个定义
        # 适应度、技能因子、任务排名(升序)、
        # 张量适应度(最优任务排名的倒数，则最大值为1)、全局最优因子(即在所有任务都是最优的个体)
        for i in range(self.pop_size):
            new_pop = {}
            new_pop['x'] = np.random.random(self.max_d)
            new_pop['fit'] = self.taskf1.function(new_pop['x'])
            new_pop['transfer'] = 0
            self.fes1 += 1
            self.task1_pop.append(new_pop)
            if new_pop['fit'] < self.task1_best:
                self.task1_best = copy.copy(new_pop['fit'])
                self.task1_best_position = copy.copy(new_pop['x'])
        for i in range(self.pop_size):
            new_pop = {}
            new_pop['x'] = np.random.random(self.max_d)
            new_pop['fit'] = self.taskf2.function(new_pop['x'])
            new_pop['transfer'] = 0
            self.fes2 += 1
            self.task2_pop.append(new_pop)
            if new_pop['fit'] < self.task2_best:
                self.task2_best = copy.copy(new_pop['fit'])
                self.task2_best_position = copy.copy(new_pop['x'])
        #对种群进行适应度排序，方便后面学习
        self.task1_pop.sort(key = lambda x:x['fit'])
        self.task2_pop.sort(key = lambda x:x['fit'])
        self.record_allbest_t1.append(copy.copy(self.task1_best))
        self.record_allbest_t2.append(copy.copy(self.task2_best))
        self.gen+=1


    def DDPM_init(self):
        self.device = get_device()
        self.pop1_model = Unet_fc(self.max_d).to(self.device)
        self.pop2_model = Unet_fc(self.max_d).to(self.device)
        self.pop1_diffusion = pop_diffusion(self.pop1_model, self.max_d,time_steps=10,loss_type='l2').to(self.device)
        self.pop2_diffusion = pop_diffusion(self.pop2_model, self.max_d, time_steps=10, loss_type='l2').to(self.device)
        self.pop1_train_ddpm = Trainer(self.pop1_diffusion,self.max_d,self.net_info['batch_size'],self.net_info['lr'],self.net_info['weight_decay'],self.net_info['num_epochs'],pop1_pop2_dataloader=self.task1_dataloader,device=self.device)
        self.pop2_train_ddpm = Trainer(self.pop2_diffusion,self.max_d,self.net_info['batch_size'],self.net_info['lr'],self.net_info['weight_decay'],self.net_info['num_epochs'],pop1_pop2_dataloader=self.task2_dataloader,device=self.device)

    def pop_data_construct(self):
        self.archive_pop1 = np.zeros((self.pop_size, self.max_d))
        self.archive_pop1_fit = np.zeros(self.pop_size)
        self.archive_pop1_label = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2 = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2_label = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2_fit = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            self.archive_pop1[i,:self.task1_d] = self.task_bound1[0]+(self.task_bound1[1]-self.task_bound1[0])*copy.copy(self.task1_pop[i]['x'][:self.task1_d])
            self.archive_pop2[i,:self.task2_d] = self.task_bound2[0] + (self.task_bound2[1] - self.task_bound2[0]) * copy.copy(self.task2_pop[i]['x'][:self.task2_d])

        self.task1_dataset = MTO_data(self.archive_pop1,self.archive_pop2)
        self.task2_dataset = MTO_data(self.archive_pop2,self.archive_pop1)
        self.task1_dataloader = DataLoader(self.task1_dataset, batch_size=self.net_info['batch_size'], shuffle=True)
        self.task2_dataloader = DataLoader(self.task2_dataset, batch_size=self.net_info['batch_size'], shuffle=True)
    def transform2list(self,pop,fit,d):
        pop_list = []
        pop_size = pop.shape[0]
        for i in range(pop_size):
            op = {}
            op['x'] = pop[i]
            op['fit'] = fit[i]
            op['transfer'] = 1
            pop_list.append(op)
        return pop_list
    def transfer_pop(self,id):
        self.pop_data_construct()
        if id ==0:
            transfer_pop1_fit = np.zeros(self.transfer_num)
            transfer_pop1 = self.pop1_train_ddpm.test_from_otherpop(self.archive_pop2[:self.transfer_num])
            transfer_pop1 = (transfer_pop1 - self.task_bound1[0]) / (self.task_bound1[1] - self.task_bound1[0])
            transfer_pop1 = np.clip(transfer_pop1, 0, 1)
            for i in range(self.transfer_num):
                transfer_pop1_fit[i] = self.taskf1.function(transfer_pop1[i])
                if transfer_pop1_fit[i] < self.task1_best:
                    self.task1_best = copy.copy(transfer_pop1_fit[i])
                self.fes1 += 1
                if (self.fes2 + self.fes1) % 200 == 0:
                    self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
                    self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
                if (self.fes1+self.fes2) >= self.MAXFES:
                    break
            for i in range(self.transfer_num):
                if transfer_pop1_fit[i] == 0:
                    transfer_pop1_fit[i] = float('inf')
            transfer_pop1 = self.transform2list(transfer_pop1, transfer_pop1_fit, self.task1_d)
            return transfer_pop1
        elif id ==1:
            transfer_pop2_fit = np.zeros(self.transfer_num)
            transfer_pop2 = self.pop2_train_ddpm.test_from_otherpop(self.archive_pop1[:self.transfer_num])
            transfer_pop2 = (transfer_pop2 - self.task_bound2[0]) / (self.task_bound2[1] - self.task_bound2[0])
            transfer_pop2 = np.clip(transfer_pop2, 0, 1)

            for i in range(self.transfer_num):
                transfer_pop2_fit[i] = self.taskf2.function(transfer_pop2[i])
                if transfer_pop2_fit[i] < self.task2_best:
                    self.task2_best = copy.copy(transfer_pop2_fit[i])
                self.fes2 += 1
                if (self.fes2+self.fes1) % 200 == 0:
                    self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
                    self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
                if (self.fes1+self.fes2) >= self.MAXFES:
                    break
            for i in range(self.transfer_num):
                if transfer_pop2_fit[i] ==0:
                    transfer_pop2_fit[i] = float('inf')
            transfer_pop2 = self.transform2list(transfer_pop2, transfer_pop2_fit, self.task2_d)
            return transfer_pop2
    def transfer_pop_count(self,pop):
        transfer_num = 0
        for i in range(len(pop)):
            if pop[i]['transfer'] == 1:
                transfer_num+=1
                pop[i]['transfer'] =0
        return pop, transfer_num



    def offspring_gen(self,pop1,pop2,op1,op2):
        offspring1 = []
        offspring2 = []
        len_pop1 = len(pop1)
        len_pop2 = len(pop2)
        if op1 == 'GA':
            for i in range(len_pop1//2):
                ca = {}
                cb = {}
                ca['transfer']=0
                cb['transfer']=0
                r1,r2 = np.random.randint(0,len_pop1,size=2)
                orand = np.random.rand()
                if orand<self.task1_rmp:
                    r2 = np.random.randint(0,len_pop2)
                    ca['x'], cb['x'] = SBX_crossover(pop1[r1]['x'], pop2[r2]['x'], self.max_d, 10, [0, 1])
                else:

                    ca['x'],cb['x'] = SBX_crossover(pop1[r1]['x'],pop1[r2]['x'],self.max_d,10,[0,1])
                ca['x'] = poly_mutation(ca['x'], self.max_d, 5, [0, 1])
                cb['x'] = poly_mutation(cb['x'], self.max_d, 5, [0, 1])

                if orand<self.task1_rmp:
                    if np.random.rand()<0.5:
                        ca['fit'] = self.taskf1.function(ca['x'])
                        self.fes1+=1
                        offspring1.append(ca)
                        if ca['fit'] < self.task1_best:
                            self.task1_best = copy.copy(ca['fit'])
                    else:
                        ca['fit'] = self.taskf2.function(ca['x'])
                        self.fes2 += 1
                        offspring2.append(ca)
                        if ca['fit'] < self.task2_best:
                            self.task2_best = copy.copy(ca['fit'])
                    if (self.fes1 + self.fes2) % 200 == 0:
                        self.record_allbest_t1.append(copy.copy(self.task1_best))
                        self.record_allbest_t2.append(copy.copy(self.task2_best))
                    if (self.fes1 + self.fes2) >= self.MAXFES:
                        break
                    if np.random.rand()<0.5:
                        cb['fit'] = self.taskf1.function(cb['x'])
                        self.fes1+=1
                        offspring1.append(cb)
                        if cb['fit'] < self.task1_best:
                            self.task1_best = copy.copy(cb['fit'])
                    else:
                        cb['fit'] = self.taskf2.function(cb['x'])
                        self.fes2 += 1
                        offspring2.append(cb)
                        if cb['fit'] < self.task2_best:
                            self.task2_best = copy.copy(cb['fit'])
                    if (self.fes1 + self.fes2) % 200 == 0:
                        self.record_allbest_t1.append(copy.copy(self.task1_best))
                        self.record_allbest_t2.append(copy.copy(self.task2_best))
                    if (self.fes1 + self.fes2) >= self.MAXFES:
                        break
                else:
                    ca['fit'] = self.taskf1.function(ca['x'])
                    self.fes1 += 1
                    offspring1.append(ca)
                    if ca['fit'] < self.task1_best:
                        self.task1_best = copy.copy(ca['fit'])
                    if (self.fes1 + self.fes2) % 200 == 0:
                        self.record_allbest_t1.append(copy.copy(self.task1_best))
                        self.record_allbest_t2.append(copy.copy(self.task2_best))
                    if (self.fes1 + self.fes2) >= self.MAXFES:
                        break
                    cb['fit'] = self.taskf1.function(cb['x'])
                    self.fes1 += 1
                    offspring1.append(cb)
                    if cb['fit'] < self.task1_best:
                        self.task1_best = copy.copy(cb['fit'])
                    if (self.fes1 + self.fes2) % 200 == 0:
                        self.record_allbest_t1.append(copy.copy(self.task1_best))
                        self.record_allbest_t2.append(copy.copy(self.task2_best))
                    if (self.fes1 + self.fes2) >= self.MAXFES:
                        break

        else:
            for i in range(len_pop1):
                pop_id = [j for j in range(len_pop1) if j != i]
                r1, r2, r3 = np.random.choice(pop_id, 3, replace=False)
                rand = np.random.rand()
                new_pop = {}
                new_pop['transfer'] = 0
                if rand < self.task1_rmp:
                    r2,r3 = np.random.randint(0,len_pop2,size=2)
                    u = DE_rand_1(pop1[i]['x'], pop1[r1]['x'], pop2[r2]['x'], pop2[r3]['x'], self.F, self.cr,
                              self.max_d, [0, 1])
                else:
                    u = DE_rand_1(pop1[i]['x'], pop1[r1]['x'], pop1[r2]['x'], pop1[r3]['x'], self.F, self.cr,
                                  self.max_d, [0, 1])
                new_pop['x'] = copy.copy(u)
                new_pop['fit'] = self.taskf1.function(u)
                self.fes1+=1
                if new_pop['fit']<self.task1_best:
                    self.task1_best = copy.copy(new_pop['fit'])
                if (self.fes1 + self.fes2) % 200 == 0:
                    self.record_allbest_t1.append(copy.copy(self.task1_best))
                    self.record_allbest_t2.append(copy.copy(self.task2_best))
                offspring1.append(new_pop)
                if (self.fes1 + self.fes2) >= self.MAXFES:
                    break
        if op2 == 'GA':
            for i in range(len_pop2//2):
                ca = {}
                cb = {}
                ca['transfer'] = 0
                cb['transfer'] = 0
                r1,r2 = np.random.randint(0,len_pop2,size=2)
                orand = np.random.rand()
                if orand<self.task2_rmp:
                    r2 = np.random.randint(0,len_pop1)
                    ca['x'], cb['x'] = SBX_crossover(pop2[r1]['x'], pop1[r2]['x'], self.max_d, 10, [0, 1])
                else:

                    ca['x'],cb['x'] = SBX_crossover(pop2[r1]['x'],pop2[r2]['x'],self.max_d,10,[0,1])
                ca['x'] = poly_mutation(ca['x'], self.max_d, 5, [0, 1])
                cb['x'] = poly_mutation(cb['x'], self.max_d, 5, [0, 1])

                if orand<self.task2_rmp:
                    if np.random.rand()<0.5:
                        ca['fit'] = self.taskf1.function(ca['x'])
                        self.fes1+=1
                        offspring1.append(ca)
                        if ca['fit'] < self.task1_best:
                            self.task1_best = copy.copy(ca['fit'])
                    else:
                        ca['fit'] = self.taskf2.function(ca['x'])
                        self.fes2 += 1
                        offspring2.append(ca)
                        if ca['fit'] < self.task2_best:
                            self.task2_best = copy.copy(ca['fit'])
                    if (self.fes1 + self.fes2) % 200 == 0:
                        self.record_allbest_t1.append(copy.copy(self.task1_best))
                        self.record_allbest_t2.append(copy.copy(self.task2_best))
                    if (self.fes1 + self.fes2) >= self.MAXFES:
                        break
                    if np.random.rand()<0.5:
                        cb['fit'] = self.taskf1.function(cb['x'])
                        self.fes1+=1
                        offspring1.append(cb)
                        if cb['fit'] < self.task1_best:
                            self.task1_best = copy.copy(cb['fit'])
                    else:
                        cb['fit'] = self.taskf2.function(cb['x'])
                        self.fes2 += 1
                        offspring2.append(cb)
                        if cb['fit'] < self.task2_best:
                            self.task2_best = copy.copy(cb['fit'])
                    if (self.fes1 + self.fes2) % 200 == 0:
                        self.record_allbest_t1.append(copy.copy(self.task1_best))
                        self.record_allbest_t2.append(copy.copy(self.task2_best))
                    if (self.fes1 + self.fes2) >= self.MAXFES:
                        break
                else:
                    ca['fit'] = self.taskf2.function(ca['x'])
                    self.fes2 += 1
                    offspring2.append(ca)
                    if ca['fit'] < self.task2_best:
                        self.task2_best = copy.copy(ca['fit'])
                    if (self.fes1 + self.fes2) % 200 == 0:
                        self.record_allbest_t1.append(copy.copy(self.task1_best))
                        self.record_allbest_t2.append(copy.copy(self.task2_best))
                    if (self.fes1 + self.fes2) >= self.MAXFES:
                        break
                    cb['fit'] = self.taskf2.function(cb['x'])
                    self.fes2 += 1
                    offspring2.append(cb)
                    if cb['fit'] < self.task2_best:
                        self.task2_best = copy.copy(cb['fit'])
                    if (self.fes1 + self.fes2) % 200 == 0:
                        self.record_allbest_t1.append(copy.copy(self.task1_best))
                        self.record_allbest_t2.append(copy.copy(self.task2_best))
                    if (self.fes1 + self.fes2) >= self.MAXFES:
                        break
        else:
            for i in range(len_pop2):
                pop_id = [j for j in range(len_pop2) if j != i]
                r1, r2, r3 = np.random.choice(pop_id, 3, replace=False)
                orand = np.random.rand()
                new_pop = {}
                new_pop['transfer'] = 0
                if orand < self.task2_rmp:
                    r2,r3 = np.random.randint(0,len_pop1,size=2)
                    u = DE_rand_1(pop2[i]['x'], pop2[r1]['x'], pop1[r2]['x'], pop1[r3]['x'], self.F, self.cr,
                              self.max_d, [0, 1])
                else:
                    u = DE_rand_1(pop2[i]['x'], pop2[r1]['x'], pop2[r2]['x'], pop2[r3]['x'], self.F, self.cr,
                                  self.max_d, [0, 1])
                new_pop['x'] = copy.copy(u)
                new_pop['fit'] = self.taskf2.function(u)
                if new_pop['fit']<self.task2_best:
                    self.task2_best = copy.copy(new_pop['fit'])
                self.fes2+=1
                if (self.fes1 + self.fes2) % 200 == 0:
                    self.record_allbest_t1.append(copy.copy(self.task1_best))
                    self.record_allbest_t2.append(copy.copy(self.task2_best))
                offspring2.append(new_pop)
                if (self.fes1 + self.fes2) >= self.MAXFES:
                    break
        return offspring1,offspring2
    def update_qtable(self,state1,nextstate1,reward1,state2,nextstate2,reward2):
        self.qtable1[state1][self.task1_action] = self.qtable1[state1][self.task1_action]+self.alpha*(reward1+self.beta*np.max(self.qtable1[nextstate1])-self.qtable1[state1][self.task1_action])
        self.qtable2[state2][self.task2_action] = self.qtable2[state2][self.task2_action] + self.alpha * (
                    reward2 + self.beta * np.max(self.qtable2[nextstate2]) - self.qtable2[state2][
                self.task2_action])
    def rlmfea_optimize(self):
        task1_nowstate = 0
        task2_nowstate = 0
        task1_nextstate = -1
        task2_nextstate = -1
        transfer_interval = [1,1]
        while (self.fes1+self.fes2)<self.MAXFES:
            if self.gen==self.train_interval:
                self.pop_data_construct()
                self.DDPM_init()
                self.pop1_train_ddpm.train()
                self.pop2_train_ddpm.train()
                self.transfer = True
                transfer_interval = [int(self.KT_interval[0]), int(self.KT_interval[1])]
            if self.gen>self.train_interval and (self.gen-self.train_interval)%self.train_interval == 0:
                self.pop_data_construct()
                self.pop1_train_ddpm.pop1_pop2_dataloader = self.task1_dataloader
                self.pop2_train_ddpm.pop1_pop2_dataloader = self.task2_dataloader
                self.pop2_train_ddpm.train()
                self.pop1_train_ddpm.train()
            if self.gen>=self.train_interval and int(self.KT_interval[0])==transfer_interval[0]:
                self.transfer_pop1= self.transfer_pop(0)
                self.task1_pop = self.task1_pop+self.transfer_pop1
                self.transfer1 = True
                transfer_interval[0] = 0
                self.task1_pop.sort(key=lambda x: x['fit'])
                if (self.fes1 + self.fes2) >= self.MAXFES:
                    break
            if self.gen>=self.train_interval and int(self.KT_interval[1])==transfer_interval[1]:
                self.transfer_pop2 = self.transfer_pop(1)
                self.task2_pop = self.task2_pop + self.transfer_pop2
                self.transfer2 = True
                transfer_interval[1] = 0
                self.task2_pop.sort(key=lambda x: x['fit'])
                if (self.fes1+self.fes2)>=self.MAXFES:
                    break
            task1_operator,task2_operator = self.operator_assign()
            parent_best = [self.task1_pop[0]['fit'],self.task2_pop[0]['fit']]
            off1, off2 = self.offspring_gen(copy.deepcopy(self.task1_pop),copy.deepcopy(self.task2_pop),task1_operator,task2_operator)
            if (self.fes1 + self.fes2) >= self.MAXFES:
                break
            off_best = [copy.copy(off1[0]['fit']),copy.copy(off2[0]['fit'])]

            self.task1_pop = self.task1_pop + off1
            self.task2_pop = self.task2_pop + off2
            self.task1_pop.sort(key=lambda x: x['fit'])
            self.task2_pop.sort(key=lambda x: x['fit'])
            self.task1_pop = self.task1_pop[:self.pop_size]
            self.task2_pop = self.task2_pop[:self.pop_size]
            if off_best[0]<parent_best[0]:
                task1_nextstate = 0
                reward1 = 10
            elif off_best[0]==parent_best[0]:
                task1_nextstate=1
                reward1 = 5
            else:
                task1_nextstate=2
                reward1 = 0

            if off_best[1]<parent_best[1]:
                task2_nextstate = 0
                reward2 = 10
            elif off_best[1]==parent_best[1]:
                task2_nextstate=1
                reward2 = 5
            else:
                task2_nextstate=2
                reward2 = 0
            self.update_qtable(task1_nowstate,task1_nextstate,reward1,task2_nowstate,task2_nextstate,reward2)
            task1_nowstate = copy.copy(task1_nextstate)
            task2_nowstate = copy.copy(task2_nextstate)

            prob_action1 = np.array([np.exp(self.qtable1[task1_nowstate][i]) for i in range(0,7)])
            prob_action2 = np.array([np.exp(self.qtable2[task2_nowstate][i]) for i in range(0, 7)])
            prob_action1 = prob_action1/np.sum(prob_action1)
            prob_action2 = prob_action2 / np.sum(prob_action2)
            prob_action1 = np.cumsum(prob_action1)
            prob_action2 = np.cumsum(prob_action2)
            self.task1_action = np.where(prob_action1 > np.random.rand())[0][0]
            self.task2_action = np.where(prob_action2 > np.random.rand())[0][0]
            self.task1_rmp = copy.copy(self.rmp_choices[self.task1_action])
            self.task2_rmp = copy.copy(self.rmp_choices[self.task2_action])

            if  self.transfer1:
                self.transfer1 = False
            if self.transfer2:
                self.transfer2 = False
            transfer_interval[0] = transfer_interval[0] + 1
            transfer_interval[1] = transfer_interval[1] + 1
            self.gen+=1
            if (self.fes1+self.fes2)>=self.MAXFES:
                break
        best_1 = self.task1_best
        best_2 = self.task2_best
        return best_1,best_2,self.record_allbest_t1,self.record_allbest_t2



task_fc = ['CIHS','CIMS','CILS','PIHS','PIMS','PILS','NIHS','NIMS','NILS']
task_2020 = [Benchmark1(),Benchmark2(),Benchmark3(),Benchmark4(),Benchmark5(),Benchmark6(),Benchmark7(),Benchmark8(),Benchmark9(),Benchmark10()]
task_2020_name = ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5','Benchmark6','Benchmark7','Benchmark8','Benchmark9','Benchmark10']
task_bound = {'CIHS':{'task1':[-100,100],'task2':[-50,50]},'CIMS':{'task1':[-50,50],'task2':[-50,50]},\
              'CILS':{'task1':[-50,50],'task2':[-500,500]},'PIHS':{'task1':[-50,50],'task2':[-100,100]},\
              'PIMS':{'task1':[-50,50],'task2':[-50,50]},'PILS':{'task1':[-50,50],'task2':[-0.5,0.5]},\
              'NIHS':{'task1':[-50,50],'task2':[-50,50]},'NIMS':{'task1':[-100,100],'task2':[-0.5,0.5]},\
              'NILS':{'task1':[-50,50],'task2':[-500,500]},}
task_d = [[50,50],[50,50],[50,50],[50,50],[50,50],[50,25],[50,50],[50,50],[50,50]]

net_info = {}
net_info['num_epochs'] = 100
net_info['lr'] = 1e-2
net_info['batch_size'] = 50
net_info['weight_decay'] = 1e-5

test_task = ['WCCI2020MTSO']
if __name__ == '__main__':
    for test in test_task:
        if test == 'WCCI2020MTSO':
            record_matrix = np.zeros((30,len(task_2020)*2))
            filename_with_suffix = __file__
            filename_without_suffix = os.path.basename(filename_with_suffix)
            filename = os.path.splitext(filename_without_suffix)[0]+'_5_25_50'+test
            ori_path = './independent_data/' + filename
            xls_savepath = './record_data/' + filename + '.xlsx'
            if not os.path.exists(ori_path):
                os.makedirs(ori_path, exist_ok=True)
            task_num = len(task_2020)
            mfea_record = openpyxl.Workbook()
            sheet = mfea_record.active
            sheet['A1'] = 'test_task'
            sheet['B1'] = 'task1'
            sheet['C1'] = 'task2'
            best_avg_array = np.zeros((task_num, 2))
            for i in range(task_num):
                task1_20_bestall = []
                task2_20_bestall = []
                path = ori_path+'/' + task_2020_name[i] + '.txt'
                task1,task2 = task_2020[i]
                task1_bound = [task1.Low, task1.High]
                task2_bound = [task2.Low, task2.High]
                task_b = {}
                task_b['task1'] = task1_bound
                task_b['task2'] = task2_bound
                for j in range(30):
                    rlmfea = RLMFEA(50,50,task1,50,task2,50,1e+05,5,50,25,net_info,task_b)
                    rlmfea.pop_init()
                    best_t1,best_t2,bestall_t1,bestall_t2 = rlmfea.rlmfea_optimize()
                    task1_20_bestall.append(bestall_t1)
                    task2_20_bestall.append(bestall_t2)
                    record_matrix[j][i*2] = best_t1
                    record_matrix[j][(i * 2+1)] = best_t2
                    best_avg_array[i][0] += best_t1
                    best_avg_array[i][1] += best_t2
                    f_file = open(path, 'a+')
                    f_file.write(f'{best_t1} {best_t2}\n')
                    f_file.close()
                task_xlsx_path = ori_path+'/' + task_2020_name[i] + '.xlsx'
                task1_20_bestall = np.array(task1_20_bestall)
                task2_20_bestall = np.array(task2_20_bestall)
                task1_20_bestall = pd.DataFrame(task1_20_bestall)
                task2_20_bestall = pd.DataFrame(task2_20_bestall)
                with pd.ExcelWriter(task_xlsx_path,engine='openpyxl') as writer:
                    task1_20_bestall.to_excel(writer,index=False, sheet_name='Task1')
                    task2_20_bestall.to_excel(writer,index=False, sheet_name='Task2')
            best_avg_array /= 30
            for i in range(task_num):
                sheet.append([task_2020_name[i], best_avg_array[i][0], best_avg_array[i][1]])
            mfea_record.save(xls_savepath)
            record_matrix = pd.DataFrame(record_matrix)
            pd_path = ori_path+'/' + 'WCCI2020MTSO.xlsx'
            record_matrix.to_excel(pd_path,index=False)