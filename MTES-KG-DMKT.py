import math
import openpyxl
import numpy as np
import os
import copy as cp
from scipy.linalg import eigh
from MLP_diffusion_model import *
from evo_operator import *
import random
from tasks import *
import pandas as pd
import argparse
from Data_construct import MTO_data
import torch
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


class DDPM_MTO:
    def __init__(self,popsize,taskf1, task1_d, taskf2, task2_d, KT_interval,train_interval,transfer_num, maxgen, net_info,task_bound):
        self.taskf = [taskf1, taskf2]
        self.pop_size = popsize
        self.max_d = max(task1_d, task2_d)
        self.MAXgen = maxgen
        self.tau0 = 2
        self.alpha = 0.5
        self.adjGap = 50
        self.sigma0 = 0.3
        self.mu = round(popsize / 2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights ** 2)
        self.chiN = np.sqrt(self.max_d) * (1 - 1 / (4 * self.max_d) + 1 / (21 * self.max_d ** 2))
        self.hth = (1.4 + 2 / (self.max_d + 1)) * self.chiN
        self.task1_d = task1_d
        self.task2_d = task2_d
        self.tasks_best = [float('inf'), float('inf')]
        self.gen = 0
        self.KT_interval = KT_interval
        self.archive_pop1 = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2 = np.zeros((self.pop_size, self.max_d))
        self.net_info = net_info
        self.transfer_num = transfer_num
        self.task_bound1 = task_bound['task1']
        self.task_bound2 = task_bound['task2']
        self.MAXfes = 1e05
        self.fes = [0, 0]
        self.train_interval = train_interval
        self.tasks_best_all = [[], []]


    def pop_init(self):
        self.tasks_pop = [[],[]]
        for k in range(len(self.taskf)):
            for i in range(self.pop_size):
                self.tasks_pop[k].append(dict())
        # 初始化任务参数
        self.params = {
            'cs': [], 'damps': [], 'cc': [], 'c1': [], 'cmu': [],
            'mDec': [], 'ps': [], 'pc': [], 'B': [], 'D': [], 'C': [],
            'invsqrtC': [], 'sigma': [], 'eigenFE': [], 'tau': [],
            'mStep': [], 'numExS': [], 'sucExS': [], 'record_tau': [],
        }

        for t in range(len(self.taskf)):
            # 初始化CMA参数
            cs = (self.mueff + 2) / (self.max_d + self.mueff + 5)
            damps = 1 + cs + 2 * max(np.sqrt((self.mueff - 1) / (self.max_d + 1)) - 1, 0)
            cc = (4 + self.mueff / self.max_d) / (4 + self.max_d + 2 * self.mueff / self.max_d)
            c1 = 2 / ((self.max_d + 1.3) ** 2 + self.mueff)
            cmu = min(1 - c1,
                      2 * (self.mueff - 2 + 1 / self.mueff) / ((self.max_d + 2) ** 2 + 2 * self.mueff / 2))

            # 存储参数
            self.params['cs'].append(cs)
            self.params['damps'].append(damps)
            self.params['cc'].append(cc)
            self.params['c1'].append(c1)
            self.params['cmu'].append(cmu)

            # 初始化分布参数
            self.params['mDec'].append(np.mean(np.random.uniform(0, 1, (self.pop_size, self.max_d)), axis=0))
            self.params['ps'].append(np.zeros(self.max_d)) #矩阵
            self.params['pc'].append(np.zeros(self.max_d))#矩阵
            self.params['B'].append(np.eye(self.max_d))#矩阵
            self.params['D'].append(np.ones(self.max_d))#矩阵
            self.params['C'].append(
                self.params['B'][t] @ np.diag(self.params['D'][t] ** 2) @ self.params['B'][t].T)
            self.params['invsqrtC'].append(
                self.params['B'][t] @ np.diag(1 / self.params['D'][t]) @ self.params['B'][t].T)
            self.params['sigma'].append(self.sigma0)
            self.params['eigenFE'].append(0)
            self.params['mStep'].append(0)
            self.params['numExS'].append([])
            self.params['sucExS'].append([])
            self.params['tau'].append(self.tau0)
            self.params['record_tau'].append([self.tau0])

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
        self.archive_pop2 = np.zeros((self.pop_size, self.max_d))
        for i in range(self.pop_size):
            self.archive_pop1[i,:self.task1_d] = self.task_bound1[0]+(self.task_bound1[1]-self.task_bound1[0])*copy.copy(self.tasks_pop[0][i]['x'][:self.task1_d])
            self.archive_pop2[i,:self.task2_d] = self.task_bound2[0] + (self.task_bound2[1] - self.task_bound2[0]) * copy.copy(self.tasks_pop[1][i]['x'][:self.task2_d])


        self.task1_dataset = MTO_data(self.archive_pop1,self.archive_pop2)
        self.task2_dataset = MTO_data(self.archive_pop2,self.archive_pop1)
        self.task1_dataloader = DataLoader(self.task1_dataset, batch_size=self.net_info['batch_size'], shuffle=True)
        self.task2_dataloader = DataLoader(self.task2_dataset, batch_size=self.net_info['batch_size'], shuffle=True)

    def optimize_tasks(self):
        while self.fes[0]+self.fes[1] < self.MAXfes:
            for k in range(len(self.tasks_pop)):
                # 精英选择策略
                self.tasks_pop[k] = self.tasks_pop[k][:self.pop_size]
            self.old_sample = copy.deepcopy(self.tasks_pop)
            self.DMKT()
            for k in range(len(self.tasks_pop)):
                for i in range(self.pop_size):
                    #生成正态分布随机向量
                    z = np.random.randn(self.max_d)
                    self.tasks_pop[k][i]['x'] = self.params['mDec'][k]+self.params['sigma'][k]*\
                                                (self.params['B'][k]@(self.params['D'][k]*z))
                    self.tasks_pop[k][i]['x'] = np.clip(self.tasks_pop[k][i]['x'], 0, 1)
                #计算平均步长
                decs = np.array([ind['x'] for ind in self.tasks_pop[k]])
                dists = np.linalg.norm(decs-self.params['mDec'][k],axis=1)
                self.params['mStep'][k] = np.mean(dists)

            self.Sample_external_solutions()


            #过了50代开始DKT
            #进行参数更新
            self.cma_esparamters_update()
            self.gen+=1
        return self.tasks_best[0],self.tasks_best_all[0],self.tasks_best[1],self.tasks_best_all[1]

    def Sample_external_solutions(self):
        #从外部采样
        for i in range(len(self.taskf)):
            tasks = list(range(0,len(self.taskf)))
            tasks.remove(i)
            k = np.random.choice(tasks)
            for j in range(self.params['tau'][i]):
                if self.gen<2:
                    z = np.random.randn(self.max_d)
                    sample_dec = self.params['mDec'][i] + self.params['sigma'][i] * \
                                                (self.params['B'][i] @ (self.params['D'][i] * z))
                    sample_dec = np.clip(sample_dec, 0, 1)
                    new_sample={'x':sample_dec}
                    self.tasks_pop[i].append(new_sample)
                    continue
                if np.random.rand()<self.alpha:
                    # DoS: 最优域知识引导的外部采样
                    z = np.random.randn(self.max_d)
                    sample_dec = self.params['mDec'][k] + self.params['sigma'][k] * \
                                 (self.params['B'][k] @ (self.params['D'][k] * z))
                    vec = sample_dec - self.params['mDec'][i]

                    norm_vec = np.linalg.norm(vec)

                    if norm_vec < self.params['mStep'][i]:
                        sample_dec = np.clip(sample_dec, 0, 1)
                        new_sample = {'x':sample_dec}
                        self.tasks_pop[i].append(new_sample)
                    else:
                        unit_vec = vec / (norm_vec)
                        new_dec = self.params['mDec'][i] + unit_vec * self.params['mStep'][i]
                        new_sample = np.clip(new_dec, 0, 1)
                        new_sample = {'x': new_dec}
                        self.tasks_pop[i].append(new_sample)
                else:
                    # SaS: 函数形状知识引导的外部采样
                    # 选择父代个体
                    idx = list(range(self.mu))
                    remove_idx = np.random.randint(0, self.mu)
                    idx.pop(remove_idx)

                    temp_matrix = np.array([
                        self.old_sample[i][l]['x']
                        for l in idx
                    ])
                    # 计算平均向量
                    vec = np.mean([
                        self.old_sample[k][l]['x']
                        for l in idx
                    ], axis=0)
                    vec = (vec - self.params['mDec'][k]) / (self.params['sigma'][k])

                    # 应用协方差矩阵变换
                    transformed_vec = self.params['B'][i] @ (self.params['D'][i] * (
                            self.params['B'][k].T @ (vec/self.params['D'][k])))
                    new_dec = self.params['mDec'][i] + self.params['sigma'][i] * transformed_vec
                    new_dec = np.clip(new_dec, 0, 1)
                    self.tasks_pop[i].append({'x':new_dec})
    def DMKT(self):
        # 判断是否满足训练条件
        if self.gen >= self.train_interval and self.gen % self.train_interval == 0:
            # 是否是第一次训练，第一次训练则需要初始化网络，若非第一次，则更新训练数据。
            if self.gen == self.train_interval:
                self.pop_data_construct()
            else:
                self.pop_data_construct()
            self.DDPM_init()
            self.pop1_train_ddpm.pop1_pop2_dataloader = self.task1_dataloader
            self.pop2_train_ddpm.pop1_pop2_dataloader = self.task2_dataloader
            self.pop2_train_ddpm.train()
            self.pop1_train_ddpm.train()
        # 判断是否满足迁移间隔
        if self.gen >= self.train_interval and self.gen % self.KT_interval == 0:
            for i in range(len(self.taskf)):
                self.tasks_pop[i] = self.tasks_pop[i] + self.transfer_pop(i)


    def transform2list(self,pop):
        pop_list = []
        pop_size = pop.shape[0]
        for i in range(pop_size):
            op = {}
            op['x'] = pop[i]
            op['fit'] = float('inf')
            pop_list.append(op)
        return pop_list

    def transfer_pop(self,id):
        self.pop_data_construct()
        gamma = 1 + sum(self.fes) / self.MAXfes * 4
        sigma = 10 ** (-gamma)
        if id ==0:
            transfer_pop1 = self.archive_pop2[:self.transfer_num]
            temp_matrix = self.task_bound1[0]+(self.task_bound1[1]-self.task_bound1[0])*np.random.normal(loc=0, scale=sigma, size=transfer_pop1.shape)
            transfer_pop1 = transfer_pop1 + 0
            transfer_pop1 = self.pop1_train_ddpm.test_from_otherpop(transfer_pop1)
            transfer_pop1 = (transfer_pop1 - self.task_bound1[0]) / (self.task_bound1[1] - self.task_bound1[0])

            transfer_pop1 = np.clip(transfer_pop1, 0, 1)

            transfer_pop1 = self.transform2list(transfer_pop1)
            return transfer_pop1
        elif id ==1:
            transfer_pop2 = self.archive_pop1[:self.transfer_num]
            temp_matrix = self.task_bound2[0] + (self.task_bound2[1] - self.task_bound2[0]) * np.random.normal(loc=0, scale=sigma, size=transfer_pop2.shape)
            transfer_pop2 = transfer_pop2 + 0
            transfer_pop2 = self.pop1_train_ddpm.test_from_otherpop(transfer_pop2)
            transfer_pop2 = (transfer_pop2 - self.task_bound2[0]) / (self.task_bound2[1] - self.task_bound2[0])
            transfer_pop2 = np.clip(transfer_pop2, 0, 1)
            transfer_pop2 = self.transform2list(transfer_pop2)
            return transfer_pop2
    def cma_esparamters_update(self):
            #更新算法参数
            #对采样和DKT的个体进行评估
            for k in range(len(self.taskf)):
                for i in range(len(self.tasks_pop[k])):
                    self.tasks_pop[k][i]['fit'] = self.taskf[k].function(self.tasks_pop[k][i]['x'])
                    if self.tasks_pop[k][i]['fit']<self.tasks_best[k]:
                        self.tasks_best[k]= self.tasks_pop[k][i]['fit']
                    if self.fes[k]==0:
                        self.tasks_best_all[k].append(self.tasks_best[k])
                    self.fes[k]+=1
                    if sum(self.fes)%200==0:
                        for j in range(len(self.taskf)):
                            self.tasks_best_all[j].append(self.tasks_best[j])

                self.params['numExS'][k].append(self.params['tau'][k])
                self.params['sucExS'][k].append(self.find_success(k))
                #对种群根据fit进行排序
                self.tasks_pop[k].sort(key=lambda x: x['fit'])
                # 负迁移缓解（调整tau）
                if self.gen % self.adjGap == 0 and self.gen > 0:
                    start_gen = max(0, self.gen - self.adjGap)
                    numAll = sum(self.params['numExS'][k][start_gen:self.gen])
                    sucAll = sum(self.params['sucExS'][k][start_gen:self.gen])

                    if (numAll > 0 and sucAll / numAll > 0.5) or numAll == 0:
                        self.params['tau'][k] = min(self.tau0, self.params['tau'][k] + 1)
                    else:
                        self.params['tau'][k] = max(0, self.params['tau'][k] - 1)
                # 更新CMA-ES参数
                # 更新均值
                old_mDec = copy.deepcopy(self.params['mDec'][k])
                # 取前一半个体作为更新参数
                top_Dec = np.array([self.tasks_pop[k][i]['x'] for i in range(self.mu)])
                self.params['mDec'][k] = np.sum(self.weights.reshape(-1, 1) * top_Dec, axis=0)

                # 更新进化路径
                y = (self.params['mDec'][k] - old_mDec) / self.params['sigma'][k]
                invc_y = self.params['invsqrtC'][k] @ y

                # 更新ps
                self.params['ps'][k] = (1 - self.params['cs'][k]) * self.params['ps'][k] + \
                                       np.sqrt(self.params['cs'][k] * (
                                                   2 - self.params['cs'][k]) * self.mueff) * invc_y

                # 计算hsig
                gen_count = (self.fes[0] + self.fes[1] - self.pop_size * (k)) // (
                            self.pop_size * len(self.taskf)) + 1
                ps_norm = np.linalg.norm(self.params['ps'][k])
                denom = np.sqrt(1 - (1 - self.params['cs'][k]) ** (2 * gen_count))
                hsig = ps_norm / denom < self.hth

                # 更新pc
                self.params['pc'][k] = (1 - self.params['cc'][k]) * self.params['pc'][k] + \
                                       hsig * np.sqrt(
                    self.params['cc'][k] * (2 - self.params['cc'][k]) * self.mueff) * y

                # 更新协方差矩阵
                artmp = (top_Dec - old_mDec).T / self.params['sigma'][k]
                delta = (1 - hsig) * self.params['cc'][k] * (2 - self.params['cc'][k])
                weights_diag = np.diag(self.weights)

                self.params['C'][k] = (1 - self.params['c1'][k] - self.params['cmu'][k]) * self.params['C'][k] + \
                                      self.params['c1'][k] * (
                                              np.outer(self.params['pc'][k], self.params['pc'][k]) + delta *
                                              self.params['C'][k]) + \
                                      self.params['cmu'][k] * artmp @ weights_diag @ artmp.T

                # 更新步长
                self.params['sigma'][k] = self.params['sigma'][k] * np.exp(
                    self.params['cs'][k] / self.params['damps'][k] * (ps_norm / self.chiN - 1))
                # 检查分布正确性（特征分解）
                if (self.fes[0] + self.fes[1] - self.pop_size * (k)) - self.params['eigenFE'][k] > (
                        self.pop_size * len(self.taskf)) / \
                        (self.params['c1'][k] + self.params['cmu'][k]) / self.max_d / 10:

                    self.params['eigenFE'][k] = self.fes[0] + self.fes[1]
                    restart = False

                    # 检查协方差矩阵的有效性
                    if np.any(np.isnan(self.params['C'][k])) or np.any(np.isinf(self.params['C'][k])):
                        restart = True
                    else:
                        # 确保对称性
                        self.params['C'][k] = np.triu(self.params['C'][k]) + np.triu(self.params['C'][k], 1).T

                        # 特征分解
                        D2, B = eigh(self.params['C'][k])
                        if np.min(D2) < 0:
                            restart = True
                        else:
                            self.params['D'][k] = np.sqrt(D2)
                            self.params['B'][k] = B

                    # 如果需要重启
                    if restart:
                        self.params['ps'][k] = np.zeros(self.max_d)
                        self.params['pc'][k] = np.zeros(self.max_d)
                        self.params['B'][k] = np.eye(self.max_d)
                        self.params['D'][k] = np.ones(self.max_d)
                        self.params['C'][k] = np.eye(self.max_d)
                        self.params['sigma'][k] = np.clip(2 * self.params['sigma'][k], 0.01, 0.3)

                    # 更新逆平方根协方差矩阵
                    invD = np.diag(1 / self.params['D'][k])
                    self.params['invsqrtC'][k] = self.params['B'][k] @ invD @ self.params['B'][k].T

                # 记录tau
                self.params['record_tau'][k].append(self.params['tau'][k])

    def find_success(self,k):
        sort_pop = copy.deepcopy(self.tasks_pop[k])
        sort_pop.sort(key=lambda x: x['fit'])
        count = 0
        if self.params['tau'][k]>0:
            for item in self.tasks_pop[k][-self.params['tau'][k]:]:
                for i in range(self.mu):
                    if item['fit']<=sort_pop[i]['fit']:
                        count+=1
                        break
        if count==50:
            print("erro")
        return count
def record_point(mat,points,i):
    for j in range(len(mat[i])):
        mat[i][j] = points[j]

    return mat






parser = argparse.ArgumentParser(description='DDMTO参数')
parser.add_argument("--transfer_num",type=int,default=5,help="迁移个数")
parser.add_argument("--train_interval",type=int,default=30,help="网络训练间隔")
parser.add_argument("--lr",type=float,default=0.01,help="learning rate")
parser.add_argument("--ti",type=int,default=5,help="learning rate")
args = parser.parse_args()
print(args)
task_fc = ['CIHS','CIMS','CILS','PIHS','PIMS','PILS','NIHS','NIMS','NILS']
task_2020 = [Benchmark1(),Benchmark2(),Benchmark3(),Benchmark4(),Benchmark5(),Benchmark6(),Benchmark7(),Benchmark8(),Benchmark9(),Benchmark10()]
task_2020_name = ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5','Benchmark6','Benchmark7','Benchmark8','Benchmark9','Benchmark10']
task_bound = {'CIHS':{'task1':[-100,100],'task2':[-50,50]},'CIMS':{'task1':[-50,50],'task2':[-50,50]},\
              'CILS':{'task1':[-50,50],'task2':[-500,500]},'PIHS':{'task1':[-50,50],'task2':[-100,100]},\
              'PIMS':{'task1':[-50,50],'task2':[-50,50]},'PILS':{'task1':[-50,50],'task2':[-0.5,0.5]},\
              'NIHS':{'task1':[-50,50],'task2':[-50,50]},'NIMS':{'task1':[-100,100],'task2':[-0.5,0.5]},\
              'NILS':{'task1':[-50,50],'task2':[-500,500]},}
task_d = [[50,50],[50,50],[50,50],[50,50],[50,50],[50,25],[50,50],[50,50],[50,50]]
test_task = ['WCCI2020MTSO']
net_info = {}
net_info['num_epochs'] = 100
net_info['lr'] = args.lr
net_info['batch_size'] = 100
net_info['weight_decay'] = 1e-5
independent_runs = 30
if __name__ == '__main__':
    for test in test_task:
        if test == 'WCCI2020MTSO':
            record_matrix = np.zeros((30,len(task_2020)*2))
            filename_with_suffix = __file__
            filename_without_suffix = os.path.basename(filename_with_suffix)
            filename = os.path.splitext(filename_without_suffix)[0]+'tn'+str(args.transfer_num)+'tri'+str(args.train_interval)+'ti'+str(args.ti)+test
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
                task1_20_bestall = np.zeros((independent_runs,501))
                task2_20_bestall = np.zeros((independent_runs,501))
                path = ori_path+'/' + task_2020_name[i] + '.txt'
                task1,task2 = task_2020[i]
                task1_bound = [task1.Low,task1.High]
                task2_bound = [task2.Low, task2.High]
                task_b = {}
                task_b['task1'] = task1_bound
                task_b['task2'] = task2_bound
                #task1 = Tasks(task_fc[i], 1)
                #task2 = Tasks(task_fc[i], 2)
                for j in range(30):
                    ddpm_mto = DDPM_MTO(100,task1,50,task2,50,args.ti,args.train_interval,args.transfer_num,500,net_info,task_b)
                    ddpm_mto.pop_init()
                    best_t1,bestall_t1,best_t2,bestall_t2 = ddpm_mto.optimize_tasks()
                    task1_20_bestall = record_point(task1_20_bestall,bestall_t1,j)
                    task2_20_bestall = record_point(task2_20_bestall, bestall_t2, j)
                    record_matrix[j][i*2] = best_t1
                    record_matrix[j][(i * 2+1)] = best_t2
                    best_avg_array[i][0] += best_t1
                    best_avg_array[i][1] += best_t2
                    f_file = open(path, 'a+')
                    f_file.write(f'{best_t1} {best_t2}\n')
                    f_file.close()
                task_xlsx_path = ori_path+'/' + task_2020_name[i] + '.xlsx'
                task_xlsx_frequency_path = ori_path + '/' + task_2020_name[i] + '_ktfrequency.xlsx'

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