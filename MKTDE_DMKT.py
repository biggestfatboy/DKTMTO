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
class MFDE:
    def __init__(self,popsize, max_d,taskf1,task1d, taskf2,task2d, maxfes, KT_interval,train_interval,transfer_num, net_info,task_bound):
        self.MAXFES = maxfes
        self.fes1 = 0
        self.fes2 = 0
        self.pop_size = popsize
        self.max_d = max(task1d,task2d)
        self.taskf1 = taskf1
        self.task1_d = task1d
        self.task2_d = task2d
        self.taskf2 = taskf2
        self.task1_num = 0
        self.task2_num = 0
        self.F = 0.5
        self.cr = 0.6
        self.record_allbest_t1 = []
        self.record_allbest_t2 = []
        self.task1_best = float('inf')
        self.task1_best_position = None
        self.task2_best = float('inf')
        self.task2_best_position = None
        self.train_interval = train_interval
        self.net_info = net_info
        self.transfer_num = transfer_num
        self.KT_interval = [float(KT_interval),float(KT_interval)]
        self.transfer1 = False
        self.transfer2 = False
        self.task_bound1 = task_bound['task1']
        self.task_bound2 = task_bound['task2']
        self.archive_pop1 = np.zeros((self.pop_size,self.max_d))
        self.archive_pop2 = np.zeros((self.pop_size, self.max_d))
        self.gen = 0
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
    def MKT(self,source_pop,target_pop):
        source_matrix = np.zeros((self.pop_size,self.max_d))
        target_matrix = np.zeros((self.pop_size, self.max_d))
        for i in range(self.pop_size):
            source_matrix[i] = copy.copy(source_pop[i]['x'])
            target_matrix[i] = copy.copy(target_pop[i]['x'])
        source_center = np.mean(source_matrix,axis=0)
        target_center = np.mean(target_matrix,axis=0)

        source_matrix = source_matrix-source_center+target_center
        ptransfer = copy.deepcopy(target_pop)
        for i in range(self.pop_size):
            new_pop = self.transmition2pop(source_matrix[i])
            ptransfer.append(new_pop)

        return ptransfer
    def EST(self,bs,tp,id):
        best_source = copy.deepcopy(bs)
        target_pop = copy.deepcopy(tp)
        if id==0:
            target_fit = self.taskf1.function(best_source['x'])
            self.fes1+=1
            if target_fit< self.task1_best:
                self.task1_best = target_fit
        else:
            target_fit = self.taskf2.function(best_source['x'])
            self.fes2+=1
            if target_fit< self.task2_best:
                self.task2_best=target_fit
        if (self.fes1 + self.fes2) % 200 == 0:
            self.record_allbest_t1.append(copy.copy(self.task1_best))
            self.record_allbest_t2.append(copy.copy(self.task2_best))

        if target_fit<target_pop[self.pop_size-1]['fit']:
            best_source['fit'] = target_fit
            target_pop[self.pop_size-1] = best_source

        return target_pop

    def transmition2pop(self,x):
        new_pop = {}
        new_pop['x'] = x
        new_pop['transfer'] =0
        return new_pop
    def llso_offspring_gen(self,ptransfer,pop,id):
        offspring = []
        for i in range(len(pop)):
            r1 = np.random.randint(0, len(pop))
            while r1 == i:
                r1 = np.random.randint(0, len(pop))
            pop_id = [j for j in range(len(ptransfer)) if j != i and j != r1]
            r2, r3 = np.random.choice(pop_id, 2, replace=False)
            u =DE_rand_1(pop[i]['x'],pop[r1]['x'],ptransfer[r2]['x'],ptransfer[r3]['x'],self.F,self.cr,self.max_d,[0,1])
            new_pop = self.transmition2pop(u)
            if id ==0:
                new_pop['fit'] = self.taskf1.function(new_pop['x'])
                self.fes1+=1
                if new_pop['fit']<self.task1_best:
                    self.task1_best = new_pop['fit']
            else:
                new_pop['fit'] = self.taskf2.function(new_pop['x'])
                self.fes2 += 1
                if new_pop['fit']<self.task2_best:
                    self.task2_best = new_pop['fit']
            offspring.append(new_pop)
            if (self.fes1+self.fes2) % 200 == 0:
                self.record_allbest_t1.append(copy.copy(self.task1_best))
                self.record_allbest_t2.append(copy.copy(self.task2_best))

            if (self.fes1+self.fes2)>=self.MAXFES:
                return offspring

        return offspring
    def llso_optimize(self):
        transfer_interval = [1, 1]
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

            ptransfer1 = self.MKT(self.task2_pop,self.task1_pop)
            ptransfer2 = self.MKT(self.task1_pop,self.task2_pop)
            if self.gen>=self.train_interval and int(self.KT_interval[0])==transfer_interval[0]:
                self.transfer_pop1= self.transfer_pop(0)
                self.task1_pop = self.task1_pop+self.transfer_pop1
                self.transfer1 = True
                transfer_interval[0] = 0
                if (self.fes1 + self.fes2) >= self.MAXFES:
                    break
            if self.gen>=self.train_interval and int(self.KT_interval[1])==transfer_interval[1]:
                self.transfer_pop2 = self.transfer_pop(1)
                self.task2_pop = self.task2_pop + self.transfer_pop2
                self.transfer2 = True
                transfer_interval[1] = 0
                if (self.fes1+self.fes2)>=self.MAXFES:
                    break
            off_task1_pop= self.llso_offspring_gen(ptransfer1,self.task1_pop,0)
            if (self.fes1+self.fes2)>=self.MAXFES:
                break
            off_task2_pop = self.llso_offspring_gen(ptransfer2, self.task2_pop, 1)
            if (self.fes1+self.fes2)>=self.MAXFES:
                break
            self.task1_pop= self.task1_pop+off_task1_pop
            self.task2_pop = self.task2_pop+off_task2_pop
            self.task1_pop.sort(key = lambda x:x['fit'])
            self.task2_pop.sort(key = lambda x:x['fit'])
            self.task1_pop = self.task1_pop[:self.pop_size]
            self.task2_pop = self.task2_pop[:self.pop_size]

            self.task1_pop = self.EST(self.task2_pop[0],self.task1_pop,0)
            if (self.fes1+self.fes2)>=self.MAXFES:
                break
            self.task2_pop = self.EST(self.task1_pop[0], self.task2_pop,1)
            if (self.fes1+self.fes2)>=self.MAXFES:
                break

            if  self.transfer1:
                self.transfer1 = False
            if self.transfer2:
                self.transfer2 = False
            self.gen+=1
            transfer_interval[0] = transfer_interval[0] + 1
            transfer_interval[1] = transfer_interval[1] + 1
        best_1 = self.task1_best
        best_2 = self.task2_best
        return best_1,best_2,self.record_allbest_t1,self.record_allbest_t2



task_fc = ['CIHS','CIMS','CILS','PIHS','PIMS','PILS','NIHS','NIMS','NILS']
task_2020 = [Benchmark1(),Benchmark2(),Benchmark3(),Benchmark4(),Benchmark5(),Benchmark6(),Benchmark7(),Benchmark8(),Benchmark9(),Benchmark10()]
task_2020_name = ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5','Benchmark6','Benchmark7','Benchmark8','Benchmark9','Benchmark10']
task_bound = {'CIHS':{'task1':[-100,100],'tkiask2':[-50,50]},'CIMS':{'task1':[-50,50],'task2':[-50,50]},\
              'CILS':{'task1':[-50,50],'task2':[-500,500]},'PIHS':{'task1':[-50,50],'task2':[-100,100]},\
              'PIMS':{'task1':[-50,50],'task2':[-50,50]},'PILS':{'task1':[-50,50],'task2':[-0.5,0.5]},\
              'NIHS':{'task1':[-50,50],'task2':[-50,50]},'NIMS':{'task1':[-100,100],'task2':[-0.5,0.5]},\
              'NILS':{'task1':[-50,50],'task2':[-500,500]},}
task_d = [[50,50],[50,50],[50,50],[50,50],[50,50],[50,25],[50,50],[50,50],[50,50]]
net_info = {}
net_info['num_epochs'] = 100
net_info['lr'] = 1e-2
net_info['batch_size'] = 100
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
                    mfllso = MFDE(50,50,task1,50,task2,50,1e+05,5,50,25,net_info,task_b)
                    mfllso.pop_init()
                    best_t1,best_t2,bestall_t1,bestall_t2 = mfllso.llso_optimize()
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