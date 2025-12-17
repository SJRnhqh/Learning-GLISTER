import numpy as np
import time
import torch
import math
import torch.nn.functional as F
from torch.utils.data import random_split, SequentialSampler, BatchSampler
from queue import PriorityQueue
from torch import random


class SetFunctionLoader_2(object):  # GLISTER算法的核心类，实现基于泰勒展开的贪心数据选择
    def __init__(  # 初始化函数
        self,  # 类实例自身
        trainset,  # 训练数据集
        x_val,  # 验证集输入数据
        y_val,  # 验证集标签
        model,  # 神经网络模型
        loss_criterion,  # 损失函数（带归约）
        loss_nored,  # 损失函数（无归约，每个样本单独计算）
        eta,  # 学习率/步长，用于一步梯度更新
        device,  # 计算设备（CPU或GPU）
        num_channels,  # 输入数据的通道数（如MNIST为1，CIFAR10为3）
        num_classes,  # 分类任务的类别数
        batch_size,  # 批次大小
    ):
        self.trainset = trainset  # assume its a sequential loader.  # 保存训练集（假设是顺序加载器）
        self.x_val = x_val.to(device)  # 将验证集输入移动到指定设备
        self.y_val = y_val.to(device)  # 将验证集标签移动到指定设备
        self.model = model  # 保存模型引用
        self.loss = (  # 保存损失函数（带归约）
            loss_criterion  # Make sure it has reduction='none' instead of default  # 确保使用reduction='none'而不是默认值
        )
        self.loss_nored = loss_nored  # 保存无归约损失函数
        self.eta = eta  # step size for the one step gradient update  # 一步梯度更新的步长（学习率）
        # self.opt = optimizer  # 优化器（已注释）
        self.device = device  # 保存计算设备
        self.N_trn = len(trainset)  # 训练集样本总数
        self.grads_per_elem = None  # 每个训练样本的梯度（初始化为None，后续计算）
        self.theta_init = None  # 初始模型参数（初始化为None）
        self.num_channels = num_channels  # 输入通道数
        self.num_classes = num_classes  # 类别数
        self.batch_size = batch_size  # 批次大小

    def _compute_per_element_grads(self, theta_init):  # 计算每个训练样本的梯度
        self.model.load_state_dict(theta_init)  # 加载初始模型参数
        batch_wise_indices = np.array(  # 创建批次索引数组
            [
                list(  # 转换为列表
                    BatchSampler(  # 批次采样器
                        SequentialSampler(np.arange(self.N_trn)),  # 顺序采样器，采样范围[0, N_trn)
                        self.batch_size,  # 批次大小
                        drop_last=False,  # 不丢弃最后不完整的批次
                    )
                )
            ][0]  # 取第一个元素（列表的列表，取内层列表）
        )
        cnt = 0  # 计数器，用于判断是否是第一个批次
        for batch_idx in batch_wise_indices:  # 遍历每个批次索引
            inputs = torch.cat(  # 拼接张量
                [
                    self.trainset[x][0].view(  # 获取第x个训练样本的第0个元素（图像），并重塑形状
                        -1,  # 批次维度自动推断
                        self.num_channels,  # 通道数
                        self.trainset[x][0].shape[1],  # 高度
                        self.trainset[x][0].shape[2],  # 宽度
                    )
                    for x in batch_idx  # 遍历批次中的每个索引
                ],
                dim=0,  # 在第0维（批次维）上拼接
            ).type(torch.float)  # 转换为浮点类型
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])  # 获取批次中所有样本的标签（第1个元素）
            inputs, targets = inputs.to(self.device), targets.to(  # 将输入和目标移动到指定设备
                self.device, non_blocking=True  # 非阻塞传输（加速）
            )
            if cnt == 0:  # 如果是第一个批次
                with torch.no_grad():  # 禁用梯度计算（评估模式）
                    data = F.softmax(self.model(inputs), dim=1)  # 计算模型输出的softmax概率分布
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)  # 创建全零张量（批次大小×类别数）
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)  # 将标签位置设为1（one-hot编码）
                outputs = tmp_tensor  # 保存one-hot编码的标签
                cnt = cnt + 1  # 计数器加1
            else:  # 如果不是第一个批次
                cnt = cnt + 1  # 计数器加1
                with torch.no_grad():  # 禁用梯度计算
                    data = torch.cat(  # 拼接数据
                        (data, F.softmax(self.model(inputs), dim=1)), dim=0  # 将当前批次的softmax输出拼接到之前的数据上
                    )
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)  # 创建全零张量
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)  # one-hot编码
                outputs = torch.cat((outputs, tmp_tensor), dim=0)  # 拼接one-hot标签
        grads_vec = data - outputs  # 计算梯度向量：预测概率 - 真实标签（这是损失函数对logits的梯度近似）
        torch.cuda.empty_cache()  # 清空CUDA缓存，释放显存
        print("Per Element Gradient Computation is Completed")  # 打印完成信息
        self.grads_per_elem = grads_vec  # 保存每个样本的梯度

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):  # 更新验证集梯度
        self.model.load_state_dict(theta_init)  # 加载初始模型参数
        self.model.zero_grad()  # 清零梯度
        if first_init:  # 如果是首次初始化
            with torch.no_grad():  # 禁用梯度计算
                scores = F.softmax(self.model(self.x_val), dim=1)  # 计算验证集的softmax概率分布
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(  # 创建全零张量（验证集大小×类别数）
                    self.device
                )
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)  # 将验证集标签转换为one-hot编码
                grads = scores - one_hot_label  # 计算验证集梯度：预测概率 - 真实标签
        # populate the gradients in model params based on loss.  # 根据损失填充模型参数的梯度
        elif grads_currX is not None:  # 如果提供了当前已选子集的梯度
            # update params:  # 更新参数
            with torch.no_grad():  # 禁用梯度计算
                params = [param for param in self.model.parameters()]  # 获取模型所有参数
                params[-1].data.sub_(self.eta * grads_currX)  # 更新最后一层参数：θ_new = θ_old - η * grads_currX（模拟一步梯度更新）
                scores = F.softmax(self.model(self.x_val), dim=1)  # 使用更新后的参数计算验证集的softmax概率
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(  # 创建全零张量
                    self.device
                )
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)  # one-hot编码
                grads = scores - one_hot_label  # 计算更新后的验证集梯度
        self.grads_val_curr = grads.mean(dim=0)  # reset parm.grads to zero!  # 计算验证集梯度的平均值（对验证集样本求平均）

    def eval_taylor(self, grads_elem, theta_init):  # 使用泰勒展开评估单个元素的增益
        grads_val = self.grads_val_curr  # 获取当前验证集梯度
        dot_prod = 0  # 初始化点积为0
        self.model.load_state_dict(theta_init)  # 加载初始模型参数
        with torch.no_grad():  # 禁用梯度计算
            params = [param for param in self.model.parameters()]  # 获取模型所有参数
            dot_prod += torch.sum(  # 累加点积
                grads_val[0] * (params[-1].data - self.eta * grads_elem[0])  # 计算：验证集梯度 · (参数 - η * 元素梯度)，这是泰勒展开的近似
            )
        return dot_prod.data  # 返回点积结果

    def eval_taylor_modular(self, grads, theta_init):  # 使用泰勒展开评估多个元素的增益（批量计算）
        grads_val = self.grads_val_curr  # 获取当前验证集梯度
        self.model.load_state_dict(theta_init)  # 加载初始模型参数
        with torch.no_grad():  # 禁用梯度计算
            grads_tensor = torch.cat(grads, dim=0)  # 将所有元素的梯度拼接成张量
            param_update = self.eta * grads_tensor  # 计算参数更新：η * 梯度
            gains = torch.matmul(param_update, grads_val)  # 计算增益：参数更新 · 验证集梯度（矩阵乘法，批量计算所有元素的增益）
        return gains  # 返回所有元素的增益向量

    # Updates gradients of set X + element (basically adding element to X)  # 更新集合X+元素的梯度（即向X中添加元素）
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!  # 注意：这会修改输入向量！grads_X是列表！grad_e是元组！
    def _update_gradients_subset(self, grads_X, element):  # 更新子集梯度（添加元素后）
        grads_e = self.grads_per_elem[element]  # 获取要添加元素的梯度
        grads_X += grads_e  # 将元素梯度累加到子集梯度上（修改输入向量）

    # Same as before i.e full batch case! No use of dataloaders here!  # 与之前相同，即全批次情况！这里不使用数据加载器！
    # Everything is abstracted away in eval call  # 所有内容都在eval调用中抽象化
    def naive_greedy_max(self, budget, theta_init):  # 朴素贪心最大化算法（GLISTER的核心选择算法）
        start_time = time.time()  # 记录开始时间
        self._compute_per_element_grads(theta_init)  # 计算每个训练样本的梯度
        end_time = time.time()  # 记录结束时间
        print("Per Element gradient computation time is: ", end_time - start_time)  # 打印梯度计算耗时
        start_time = time.time()  # 记录开始时间
        self._update_grads_val(theta_init, first_init=True)  # 初始化验证集梯度
        end_time = time.time()  # 记录结束时间
        print(  # 打印验证集梯度计算耗时
            "Updated validation set gradient computation time is: ",
            end_time - start_time,
        )
        # Dont need the trainloader here!! Same as full batch version!  # 这里不需要训练数据加载器！！与全批次版本相同！
        numSelected = 0  # 已选择的样本数量
        grads_currX = []  # basically stores grads_X for the current greedy set X  # 存储当前贪心集合X的梯度
        greedySet = list()  # 贪心选择的样本索引列表
        remainSet = list(range(self.N_trn))  # 剩余未选择的样本索引列表（初始为所有训练样本）
        t_ng_start = time.time()  # naive greedy start time  # 贪心算法开始时间
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))  # 计算每次候选子集大小（用于加速，不是评估所有剩余样本）
        while numSelected < budget:  # 当已选择数量小于预算时循环
            # Try Using a List comprehension here!  # 尝试在这里使用列表推导式！
            t_one_elem = time.time()  # 记录选择单个元素开始时间
            subset_selected = list(  # 从剩余集合中随机选择候选子集
                np.random.choice(  # 随机选择
                    np.array(list(remainSet)), size=subset_size, replace=False  # 从剩余集合中不重复地选择subset_size个样本
                )
            )
            rem_grads = [  # 获取候选子集中每个样本的梯度
                self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0])  # 重塑梯度形状为(1, 类别数)
                for x in subset_selected  # 遍历候选子集中的每个索引
            ]
            gains = self.eval_taylor_modular(rem_grads, theta_init)  # 使用泰勒展开批量评估所有候选样本的增益
            # Update the greedy set and remaining set  # 更新贪心集合和剩余集合
            bestId = subset_selected[torch.argmax(gains)]  # 选择增益最大的样本索引
            greedySet.append(bestId)  # 将最佳样本添加到贪心集合
            remainSet.remove(bestId)  # 从剩余集合中移除最佳样本
            # Update info in grads_currX using element=bestId  # 使用bestId元素更新grads_currX信息
            if numSelected > 0:  # 如果不是第一次选择
                self._update_gradients_subset(grads_currX, bestId)  # 更新子集梯度（累加新元素的梯度）
            else:  # If 1st selection, then just set it to bestId grads  # 如果是第一次选择，则直接设置为bestId的梯度
                grads_currX = self.grads_per_elem[  # 直接赋值
                    bestId
                ]  # Making it a list so that is mutable!  # 使其成为列表以便可变！
            # Update the grads_val_current using current greedySet grads  # 使用当前贪心集合的梯度更新验证集梯度
            self._update_grads_val(theta_init, grads_currX)  # 更新验证集梯度（模拟使用当前子集训练后的模型状态）
            if numSelected % 1000 == 0:  # 每选择1000个样本打印一次
                # Printing bestGain and Selection time for 1 element.  # 打印最佳增益和选择1个元素的时间
                print(
                    "numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem
                )
            numSelected += 1  # 已选择数量加1
        print("Naive greedy total time:", time.time() - t_ng_start)  # 打印贪心算法总耗时
        return list(greedySet), grads_currX  # 返回贪心选择的样本索引列表和最终子集的梯度


class NonDeepSetFunctionLoader_2(object):
    def __init__(
        self,
        trainset,
        x_val,
        y_val,
        model,
        loss_criterion,
        loss_nored,
        eta,
        device,
        num_classes,
        batch_size,
    ):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.model = model
        self.loss = (
            loss_criterion  # Make sure it has reduction='none' instead of default
        )
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [
                list(
                    BatchSampler(
                        SequentialSampler(np.arange(self.N_trn)),
                        self.batch_size,
                        drop_last=False,
                    )
                )
            ][0]
        )
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [self.trainset[x][0].view(1, -1) for x in batch_idx], dim=0
            ).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(
                self.device, non_blocking=True
            )
            if cnt == 0:
                with torch.no_grad():
                    data = F.softmax(self.model(inputs), dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                cnt = cnt + 1
            else:
                cnt = cnt + 1
                with torch.no_grad():
                    data = torch.cat(
                        (data, F.softmax(self.model(inputs), dim=1)), dim=0
                    )
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = torch.cat((outputs, tmp_tensor), dim=0)
        grads_vec = data - outputs
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(
                    self.device
                )
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX)
                scores = F.softmax(self.model(self.x_val), dim=1)
                one_hot_label = torch.zeros(len(self.y_val), self.num_classes).to(
                    self.device
                )
                one_hot_label.scatter_(1, self.y_val.view(-1, 1), 1)
                grads = scores - one_hot_label
        self.grads_val_curr = grads.mean(dim=0)  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(
                grads_val[0] * (params[-1].data - self.eta * grads_elem[0])
            )
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print(
            "Updated validation set gradient computation time is: ",
            end_time - start_time,
        )
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while numSelected < budget:
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(
                np.random.choice(
                    np.array(list(remainSet)), size=subset_size, replace=False
                )
            )
            rem_grads = [
                self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0])
                for x in subset_selected
            ]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[
                    bestId
                ]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print(
                    "numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem
                )
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet)


class WeightedSetFunctionLoader(object):
    def __init__(
        self,
        trainset,
        x_val,
        y_val,
        facloc_size,
        lam,
        model,
        loss_criterion,
        loss_nored,
        eta,
        device,
        num_channels,
        num_classes,
        batch_size,
    ):
        self.trainset = trainset  # assume its a sequential loader.
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.facloc_size = facloc_size
        self.lam = lam
        self.model = model
        self.loss = (
            loss_criterion  # Make sure it has reduction='none' instead of default
        )
        self.loss_nored = loss_nored
        self.eta = eta  # step size for the one step gradient update
        # self.opt = optimizer
        self.device = device
        self.N_trn = len(trainset)
        self.grads_per_elem = None
        self.theta_init = None
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size

    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        batch_wise_indices = np.array(
            [
                list(
                    BatchSampler(
                        SequentialSampler(np.arange(self.N_trn)),
                        self.batch_size,
                        drop_last=False,
                    )
                )
            ][0]
        )
        cnt = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [
                    self.trainset[x][0].view(
                        -1,
                        self.num_channels,
                        self.trainset[x][0].shape[1],
                        self.trainset[x][0].shape[2],
                    )
                    for x in batch_idx
                ],
                dim=0,
            ).type(torch.float)
            targets = torch.tensor([self.trainset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(self.device), targets.to(
                self.device, non_blocking=True
            )
            if cnt == 0:
                with torch.no_grad():
                    data = F.softmax(self.model(inputs), dim=1)
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = tmp_tensor
                cnt = cnt + 1
            else:
                cnt = cnt + 1
                with torch.no_grad():
                    data = torch.cat(
                        (data, F.softmax(self.model(inputs), dim=1)), dim=0
                    )
                tmp_tensor = torch.zeros(len(inputs), self.num_classes).to(self.device)
                tmp_tensor.scatter_(1, targets.view(-1, 1), 1)
                outputs = torch.cat((outputs, tmp_tensor), dim=0)
        grads_vec = data - outputs
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        self.grads_per_elem = grads_vec

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.load_state_dict(theta_init)
        self.model.zero_grad()
        if first_init:
            with torch.no_grad():
                for i in range(10):
                    batch_scores = F.softmax(
                        self.model(
                            self.x_val[
                                (i)
                                * int((len(self.x_val)) / 10) : (i + 1)
                                * int((len(self.x_val)) / 10)
                            ]
                        ),
                        dim=1,
                    )
                    batch_one_hot_label = torch.zeros(
                        len(
                            self.y_val[
                                (i)
                                * int(len(self.x_val) / 10) : (i + 1)
                                * int(len(self.x_val) / 10)
                            ]
                        ),
                        self.num_classes,
                    ).to(self.device)
                    if i == 0:
                        scores = batch_scores
                        one_hot_label = batch_one_hot_label.scatter_(
                            1,
                            self.y_val[
                                (i)
                                * int(len(self.x_val) / 10) : (i + 1)
                                * int(len(self.x_val) / 10)
                            ].view(-1, 1),
                            1,
                        )
                    else:
                        scores = torch.cat((scores, batch_scores), dim=0)
                        one_hot_label = torch.cat(
                            (
                                one_hot_label,
                                batch_one_hot_label.scatter_(
                                    1,
                                    self.y_val[
                                        (i)
                                        * int(len(self.x_val) / 10) : (i + 1)
                                        * int(len(self.x_val) / 10)
                                    ].view(-1, 1),
                                    1,
                                ),
                            ),
                            dim=0,
                        )
                grads = scores - one_hot_label
                grads[0 : int(self.facloc_size)] = (
                    self.lam * grads[0 : int(self.facloc_size)]
                )
        # populate the gradients in model params based on loss.
        elif grads_currX is not None:
            # update params:
            with torch.no_grad():
                params = [param for param in self.model.parameters()]
                params[-1].data.sub_(self.eta * grads_currX)
                for i in range(10):
                    batch_scores = F.softmax(
                        self.model(
                            self.x_val[
                                (i)
                                * int((len(self.x_val)) / 10) : (i + 1)
                                * int((len(self.x_val)) / 10)
                            ]
                        ),
                        dim=1,
                    )
                    batch_one_hot_label = torch.zeros(
                        len(
                            self.y_val[
                                (i)
                                * int(len(self.x_val) / 10) : (i + 1)
                                * int(len(self.x_val) / 10)
                            ]
                        ),
                        self.num_classes,
                    ).to(self.device)
                    if i == 0:
                        scores = batch_scores
                        one_hot_label = batch_one_hot_label.scatter_(
                            1,
                            self.y_val[
                                (i)
                                * int(len(self.x_val) / 10) : (i + 1)
                                * int(len(self.x_val) / 10)
                            ].view(-1, 1),
                            1,
                        )
                    else:
                        scores = torch.cat((scores, batch_scores), dim=0)
                        one_hot_label = torch.cat(
                            (
                                one_hot_label,
                                batch_one_hot_label.scatter_(
                                    1,
                                    self.y_val[
                                        (i)
                                        * int(len(self.x_val) / 10) : (i + 1)
                                        * int(len(self.x_val) / 10)
                                    ].view(-1, 1),
                                    1,
                                ),
                            ),
                            dim=0,
                        )
                grads = scores - one_hot_label
                grads[0 : int(self.facloc_size)] = (
                    self.lam * grads[0 : int(self.facloc_size)]
                )
        self.grads_val_curr = grads.mean(dim=0)  # reset parm.grads to zero!

    def eval_taylor(self, grads_elem, theta_init):
        grads_val = self.grads_val_curr
        dot_prod = 0
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            params = [param for param in self.model.parameters()]
            dot_prod += torch.sum(
                grads_val[0] * (params[-1].data - self.eta * grads_elem[0])
            )
        return dot_prod.data

    def eval_taylor_modular(self, grads, theta_init):
        grads_val = self.grads_val_curr
        self.model.load_state_dict(theta_init)
        with torch.no_grad():
            grads_tensor = torch.cat(grads, dim=0)
            param_update = self.eta * grads_tensor
            gains = torch.matmul(param_update, grads_val)
        return gains

    # Updates gradients of set X + element (basically adding element to X)
    # Note that it modifies the inpute vector! Also grads_X is a list! grad_e is a tuple!
    def _update_gradients_subset(self, grads_X, element):
        grads_e = self.grads_per_elem[element]
        grads_X += grads_e

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def naive_greedy_max(self, budget, theta_init):
        start_time = time.time()
        self._compute_per_element_grads(theta_init)
        end_time = time.time()
        print("Per Element gradient computation time is: ", end_time - start_time)
        start_time = time.time()
        self._update_grads_val(theta_init, first_init=True)
        end_time = time.time()
        print(
            "Updated validation set gradient computation time is: ",
            end_time - start_time,
        )
        # Dont need the trainloader here!! Same as full batch version!
        numSelected = 0
        grads_currX = []  # basically stores grads_X for the current greedy set X
        greedySet = list()
        remainSet = list(range(self.N_trn))
        t_ng_start = time.time()  # naive greedy start time
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while numSelected < budget:
            # Try Using a List comprehension here!
            t_one_elem = time.time()
            subset_selected = list(
                np.random.choice(
                    np.array(list(remainSet)), size=subset_size, replace=False
                )
            )
            rem_grads = [
                self.grads_per_elem[x].view(1, self.grads_per_elem[0].shape[0])
                for x in subset_selected
            ]
            gains = self.eval_taylor_modular(rem_grads, theta_init)
            # Update the greedy set and remaining set
            bestId = subset_selected[torch.argmax(gains)]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            # Update info in grads_currX using element=bestId
            if numSelected > 0:
                self._update_gradients_subset(grads_currX, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_currX = self.grads_per_elem[
                    bestId
                ]  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(theta_init, grads_currX)
            if numSelected % 1000 == 0:
                # Printing bestGain and Selection time for 1 element.
                print(
                    "numSelected:", numSelected, "Time for 1:", time.time() - t_one_elem
                )
            numSelected += 1
        print("Naive greedy total time:", time.time() - t_ng_start)
        return list(greedySet), grads_currX
