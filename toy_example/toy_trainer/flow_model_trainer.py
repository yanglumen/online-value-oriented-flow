import os.path

import torch
import random
import torch.nn as nn
import numpy as np
from toy_example.flow_model.forward_process import (sample_interpolated_points,
                                                    sample_guided_interpolated_points,
                                                    sample_weighted_interpolated_points)
from trainer.trainer_util import (
    batch_to_device,
)
import torch.nn.functional as F
import matplotlib.pyplot as plt
from toy_example.toy_dataset import inf_train_gen
from toy_example.config.hyperparameter import WeightedSamplesType, FlowGuidedMode

def cycle_dataloader(argus, dataset, train_batch_size):
    random.seed(argus.seed)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True)
    while True:
        for data in dataset_loader:
            yield data
        print("Finish this epoch dataloader !!!!!!!")
        random.seed(np.random.randint(0, 9999))
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True)
        random.seed(argus.seed)

class toy_flow_trainer():
    def __init__(self, argus, model, energy_model, dataset):
        self.argus = argus
        self.model = model
        self.model.to(self.argus.device)
        if argus.flow_guided_mode == FlowGuidedMode.normal:
            self.energy_model = energy_model
        elif argus.flow_guided_mode == FlowGuidedMode.expectile_rl:
            self.energy_model, self.expectile_energy_model, self.direction_energy_model = energy_model
            self.expectile_energy_model.to(self.argus.device)
            self.direction_energy_model.to(self.argus.device)
            self.expectile_energy_optimizer = torch.optim.Adam(self.expectile_energy_model.parameters(), lr=argus.lr)
            self.direction_energy_optimizer = torch.optim.Adam(self.direction_energy_model.parameters(), lr=argus.lr)
        else:
            raise ValueError('Invalid value for argus.flow_guided_mode')
        self.energy_model.to(self.argus.device)
        self.dataset = dataset
        self.device = argus.device
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=argus.lr)
        self.energy_optimizer = torch.optim.Adam(self.energy_model.parameters(), lr=argus.lr)
        self.step = 0
        self.dataloader = cycle_dataloader(
            argus=self.argus, dataset=self.dataset, train_batch_size=argus.batch_size)
        # self.visualize_ground_truth(num_samples=20000)
        # raise NotImplementedError

    def guided_flow_train(self, batch):
        # unweighted_x_t, unweighted_t, unweighted_dx_dt = sample_interpolated_points(
        #     len(batch.datas), batch.datas.squeeze(dim=1))
        # unweighted_weights = torch.ones_like(unweighted_t)
        unweighted_x_t, unweighted_t, unweighted_dx_dt, unweighted_weights = sample_weighted_interpolated_points(
            data=batch.datas.squeeze(dim=1), energy=batch.energy.squeeze(dim=1), beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=self.argus.weighted_samples_type[0],
        )
        weighted_x_t, weighted_t, weighted_dx_dt, weighted_weights = sample_weighted_interpolated_points(
            data=batch.datas.squeeze(dim=1), energy=batch.energy.squeeze(dim=1), beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=self.argus.weighted_samples_type[1],
        )
        if self.argus.time_rescale:
            unweighted_t = unweighted_t / 2.0
            weighted_t = weighted_t / 2.0 + 0.5
        x_t = torch.cat((unweighted_x_t, weighted_x_t), dim=0)
        t = torch.cat((unweighted_t, weighted_t), dim=0)
        dx_dt = torch.cat((unweighted_dx_dt, weighted_dx_dt), dim=0)
        weights = torch.cat((unweighted_weights, weighted_weights), dim=0)
        # weights = weights / weights.sum()
        v_pred = self.model(x_t, t)
        loss = torch.mean((v_pred - dx_dt)**2 * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"flow_loss": loss.item()}

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_guided_flow_train(self, batch):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            data=batch.datas.squeeze(dim=1), energy=batch.energy.squeeze(dim=1), beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            norm_direction=False,
        )
        with torch.no_grad():
            normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
            q = self.direction_energy_model(torch.cat([x_t, normed_dx_dt], dim=-1))
            v = self.expectile_energy_model(x_t)
            u = q - v
            weights = torch.where(u > 0, self.argus.tau, (1 - self.argus.tau)).detach()

        # x_t1, t1, dx_dt1, weights1 = sample_weighted_interpolated_points(
        #     data=batch.datas.squeeze(dim=1), energy=batch.energy.squeeze(dim=1), beta=self.argus.beta,
        #     energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        #     norm_direction=True,
        # )
        # x_t2, t2, dx_dt2, weights2 = sample_weighted_interpolated_points(
        #     data=batch.datas.squeeze(dim=1), energy=batch.energy.squeeze(dim=1), beta=self.argus.beta,
        #     energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.random_direction,
        #     norm_direction=True,
        # )
        # x_t = torch.cat((x_t1, x_t2), dim=0)
        # t = torch.cat((t1, t2), dim=0)
        # dx_dt = torch.cat((dx_dt1, dx_dt2), dim=0)
        # with torch.no_grad():
        #     q = self.direction_energy_model(torch.cat([x_t, dx_dt], dim=-1))
        #     v = self.expectile_energy_model(x_t)
        #     u = q - v
        #     weights = torch.where(u > 0, tau, (1 - tau)).detach()
        pred_dx_dt = self.model(x_t, t)
        # flow_loss = weights * torch.pow(pred_dx_dt-dx_dt.detach(), 2)
        # flow_loss = weights * F.cosine_similarity(pred_dx_dt, dx_dt.detach(), dim=-1).unsqueeze(dim=-1)
        flow_loss = weights * torch.norm(pred_dx_dt-dx_dt.detach(), dim=-1, keepdim=True)
        flow_loss = flow_loss.mean()
        self.optimizer.zero_grad()
        flow_loss.backward()
        self.optimizer.step()
        return {"flow_loss": flow_loss.detach().cpu().numpy().item()}

    def flow_train(self, batch):
        if self.argus.flow_guided_mode == FlowGuidedMode.normal:
            flow_loss_info = self.guided_flow_train(batch)
        elif self.argus.flow_guided_mode == FlowGuidedMode.expectile_rl:
            flow_loss_info = self.expectile_guided_flow_train(batch)
        else:
            raise NotImplementedError
        return flow_loss_info

    def guided_energy_train(self, batch):
        x_t, _, __ = sample_interpolated_points(len(batch.datas), batch.datas.squeeze(dim=1))
        input = torch.cat((x_t, batch.datas.squeeze(dim=1)), dim=0)
        label = torch.cat((torch.zeros_like(batch.energy.squeeze(dim=1)), batch.energy.squeeze(dim=1)), dim=0)
        e_pred = self.energy_model(input)
        loss = self.loss_fn(e_pred, label)
        self.energy_optimizer.zero_grad()
        loss.backward()
        self.energy_optimizer.step()
        return {"energy_loss": loss.item()}

    def expectile_guided_energy_train(self, batch):
        x_t, t, dx_dt = sample_interpolated_points(len(batch.datas), batch.datas.squeeze(dim=1))
        e_pred = self.energy_model(batch.datas.squeeze(dim=1))
        e_loss = self.loss_fn(e_pred, batch.energy.squeeze(dim=1))
        self.energy_optimizer.zero_grad()
        e_loss.backward()
        self.energy_optimizer.step()

        dx_dt = dx_dt / torch.norm(dx_dt, dim=1, keepdim=True)
        Q_x_t = self.direction_energy_model(torch.cat([x_t, dx_dt], dim=-1))
        V_x_t = self.expectile_energy_model(x_t)
        q_loss = self.loss_fn(Q_x_t, batch.energy.squeeze(dim=1))
        v_loss = self.expectile_loss(tau=self.argus.tau, u=Q_x_t.detach() - V_x_t)

        # random_direction = torch.randn_like(dx_dt, device=self.argus.device)
        # random_direction = random_direction / torch.norm(random_direction, dim=-1, keepdim=True) * torch.norm(dx_dt, dim=-1, keepdim=True)
        # x_t_plus_1 = x_t + random_direction * t/self.argus.flow_steps
        # with torch.no_grad():
        #     energy_x_t_plus_1 = self.energy_model(x_t_plus_1).detach()
        # Q_x_t = self.direction_energy_model(torch.cat([x_t, random_direction], dim=-1))
        # V_x_t = self.expectile_energy_model(x_t)
        # q_loss = self.loss_fn(Q_x_t, energy_x_t_plus_1)
        # v_loss = self.expectile_loss(tau=self.argus.tau, u=Q_x_t.detach() - V_x_t)

        self.direction_energy_optimizer.zero_grad()
        q_loss.backward()
        self.direction_energy_optimizer.step()
        self.expectile_energy_optimizer.zero_grad()
        v_loss.backward()
        self.expectile_energy_optimizer.step()
        return {"energy_loss": e_loss.detach().cpu().numpy().item(),
                "direction_energy_loss": q_loss.detach().cpu().numpy().item(),
                "expectile_energy_loss": v_loss.detach().cpu().numpy().item()}

    def energy_train(self, batch):
        if self.argus.flow_guided_mode == FlowGuidedMode.normal:
            energy_loss_info = self.guided_energy_train(batch)
        elif self.argus.flow_guided_mode == FlowGuidedMode.expectile_rl:
            energy_loss_info = self.expectile_guided_energy_train(batch)
        else:
            raise ValueError("Unsupported energy mode")
        return energy_loss_info

    def guided_train(self, num_epochs, num_steps_per_epoch):
        for epoch in range(num_epochs):
            if epoch % 10 == 0:
                self.model.eval()
                self.visualize_sampled_data(epoch=epoch, x_record=self.argus.x_record)
                self.model.train()
            for step in range(num_steps_per_epoch):
                update_energy = True if epoch < 2 else False
                update_flow = False if epoch < 2 else True

                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                loss = {}
                if update_energy:
                    energy_loss_info = self.energy_train(batch)
                    loss.update(energy_loss_info)
                if update_flow:
                    flow_loss_info = self.flow_train(batch)
                    loss.update(flow_loss_info)
                if self.step % 200 == 0:
                    print(f"Epoch {epoch} | Step {self.step} | Loss: {loss}")
                self.step += 1

    # def flow_train(self, batch):
    #     x_t, t, dx_dt = sample_interpolated_points(len(batch.datas), batch.datas.squeeze(dim=1))  # 采样训练数据
    #     v_pred = self.model(x_t, t)  # 预测流向
    #     loss = self.loss_fn(v_pred, dx_dt)  # 计算 MSE 损失
    #     # loss = torch.mean((v_pred - dx_dt)**2)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return {"flow_loss": loss.item()}
    #
    # def train(self, num_epochs, num_steps_per_epoch):
    #     for epoch in range(num_epochs):
    #         if epoch % 10 == 0:
    #             self.visualize_sampled_data(epoch=epoch)
    #         for step in range(num_steps_per_epoch):
    #             batch = next(self.dataloader)
    #             batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
    #             loss = {}
    #             energy_loss_info = self.energy_train(batch)
    #             loss.update(energy_loss_info)
    #             flow_loss_info = self.flow_train(batch)
    #             loss.update(flow_loss_info)
    #             if self.step % 100 == 0:
    #                 print(f"Epoch {epoch} | Step {self.step} | Loss: {loss}")
    #             self.step += 1

    def sample_from_flow(self, time_start=0.0, time_end=1.0, num_samples=1000, steps=100, x_record=False):
        x = torch.randn(num_samples, 2, device=self.argus.device)  # 从高斯分布初始化
        dt = 1.0 / steps
        multi_x = []
        # for current_generation_time in range(self.argus.multi_stage_genration):
        step_index = 0
        for t in np.linspace(time_start, time_end, steps):  # 逐步演化
            # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
            t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
            v = self.model(x, t_tensor) #* self.energy_model(x) * scale
            x = x + v * dt * self.argus.flow_step_scale  # 通过 ODE 进行更新
            step_index += 1
            if x_record:
                if step_index == 40:
                    multi_x.append(x)
        return x, multi_x

    def visualize_sampled_data(self, epoch, num_samples=500, steps=100, x_record=False, picture_path='toy_example/generated_pictures'):
        plt.figure()
        generated_data = []
        multi_generated_data = []
        for __ in range(6):
            x, multi_x = self.sample_from_flow(num_samples=num_samples, steps=steps, x_record=x_record)
            generated_data.append(x.detach().cpu().numpy())
            if x_record:
                if len(multi_generated_data) == 0:
                    for __ in range(len(multi_x)):
                        multi_generated_data.append([])
                for x_i_record in range(len(multi_x)):
                    multi_generated_data[x_i_record].append(multi_x[x_i_record].detach().cpu().numpy())
        generated_data = np.concatenate(generated_data, axis=0)
        plt.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.2, s=5, c='#2d6a4f', edgecolors='none')
        if x_record:
            for x_i_record in range(len(multi_generated_data)):
                tmp_data = np.concatenate(multi_generated_data[x_i_record], axis=0)
                plt.scatter(tmp_data[:, 0], tmp_data[:, 1], alpha=0.2, s=5,
                            c=self.argus.color_list[x_i_record], edgecolors='none')
        # plt.hexbin(generated_data[:, 0], generated_data[:, 1], gridsize=50, cmap='magma', alpha=0.9)
        # plt.colorbar(label='Point Density')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.title(f"Epoch_{epoch} Generated Data")
        os.makedirs(f"{picture_path}/{self.argus.dataset}", exist_ok=True)
        plt.savefig(f"{picture_path}/{self.argus.dataset}/Epoch_{epoch}.png")

    def visualize_ground_truth(self, num_samples=10000):
        datas, energy = inf_train_gen(self.argus.dataset, batch_size=num_samples)
        plt.figure()
        plt.scatter(datas[:, 0], datas[:, 1], alpha=0.2, s=5, c='#2d6a4f', edgecolors='none')
        # plt.hexbin(datas[:, 0], datas[:, 1], gridsize=50, cmap='magma', alpha=0.9)
        # plt.colorbar(label='Point Density')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.title(f"GroundTruth Data")
        os.makedirs(f"toy_example/generated_pictures/{self.argus.dataset}", exist_ok=True)
        plt.savefig(f"toy_example/generated_pictures/{self.argus.dataset}/GroundTruth.png")

    def energy_gradient_guided_flow_train(self, batch):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            data=batch.datas.squeeze(dim=1), energy=batch.energy.squeeze(dim=1), beta=1.0, energy_model=None,
            weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        with torch.enable_grad():
            x_t.requires_grad_(True)
            pred_e = self.energy_model(x_t)
            grad_of_x_t = torch.autograd.grad(torch.sum(pred_e), x_t)[0]
        grad_of_x_t = grad_of_x_t / torch.norm(grad_of_x_t, p=2, dim=1, keepdim=True)
        # dx_dt = dx_dt / torch.norm(dx_dt, p=2, dim=1, keepdim=True)
        v_pred = self.model(x_t, t)
        loss = self.loss_fn(v_pred, grad_of_x_t.detach()) + 0.1*torch.mean((v_pred - dx_dt.detach()) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"flow_loss": loss.item()}

    def plt_energy_scat(self):
        plt.figure(figsize=(12, 12))
        x = np.linspace(-4, 4, 50)  # x 坐标
        y = np.linspace(-4, 4, 50)  # y 坐标
        X, Y = np.meshgrid(x, y)
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        data = np.vstack((X_flat, Y_flat)).T
        data = torch.tensor(data, dtype=torch.float32, device=self.argus.device).requires_grad_(True)
        V = self.energy_model(data)
        gradients = torch.autograd.grad(V.sum(), data, create_graph=True)[0]
        gradients = gradients / gradients.norm(2, dim=1, keepdim=True) * 0.2
        grad_x = gradients[:, 0].detach().cpu().numpy().reshape(X.shape)
        grad_y = gradients[:, 1].detach().cpu().numpy().reshape(Y.shape)
        plt.quiver(X, Y, grad_x, grad_y, scale=20, color='blue')  # 使用 -∇E 作为方向
        plt.savefig(f"toy_example/energy_gradient_guided_generated_pictures/scatter.png")
        V = V.detach().cpu().numpy()
        # 绘制散点图
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_flat, Y_flat, c=V, cmap='viridis', s=50, edgecolors='none')
        # 添加颜色条
        plt.colorbar(scatter, label="Value (v)")
        plt.show()

    def energy_gradient_guided_train(self, num_epochs, num_steps_per_epoch):
        for epoch in range(num_epochs):
            if epoch % 10 == 0:
                self.model.eval()
                self.visualize_sampled_data(
                    epoch=epoch, x_record=self.argus.x_record, picture_path="toy_example/energy_gradient_guided_generated_pictures")
                self.model.train()
            for step in range(num_steps_per_epoch):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                loss = {}
                energy_loss_info = self.energy_train(batch)
                loss.update(energy_loss_info)
                if self.step >= 20000:
                    # self.plt_energy_scat()
                    # raise NotImplementedError
                    flow_loss_info = self.energy_gradient_guided_flow_train(batch)
                    loss.update(flow_loss_info)
                if self.step % 100 == 0:
                    print(f"Epoch {epoch} | Step {self.step} | Loss: {loss}")
                self.step += 1

