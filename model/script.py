utf-8
import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np


print("" * 50)

print("")

print("" * 50)



class CRNSharedFeatureExtractor(nn.Module):

    """
    CRN 共享特征提取器: 多层 MLP，接受多个模态特征，输出共享语义表示。
    对应论文公式 (8): s_v = fsha(v, t; θsha)
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=256, n_layers=20):

        super(CRNSharedFeatureExtractor, self).__init__()


        layers = []

        current_dim = input_dim




        for i in range(n_layers):

            if i == n_layers - 1:



                next_dim = output_dim

            else:

                next_dim = hidden_dim


            layers.append(nn.Linear(current_dim, next_dim))


            if i < n_layers - 1:



                layers.append(nn.LayerNorm(next_dim))

                layers.append(nn.ReLU())

                layers.append(nn.Dropout(0.1))


            current_dim = next_dim


        self.net = nn.Sequential(*layers)


    def forward(self, *modalities):

        ""



        combined = torch.cat(modalities, dim=-1)



        return self.net(combined)



class CRNModalityAwareReconstructor(nn.Module):

    """
    CRN 模态感知重建器: 多层 MLP，结合共享语义 + 模态特定描述符嵌入。
    对应论文公式 (10): â = fspec([P̂M, s_v, s_t]; θspec)
    """

    def __init__(self, shared_dim=256, target_dim=512, descriptor_dim=128, n_layers=10):

        super(CRNModalityAwareReconstructor, self).__init__()

        self.target_dim = target_dim

        self.shared_dim = shared_dim

        self.n_layers = n_layers




        proj_layers = []

        current_dim = shared_dim

        for i in range(3):



            proj_layers.append(nn.Linear(current_dim, target_dim))

            proj_layers.append(nn.LayerNorm(target_dim))

            proj_layers.append(nn.ReLU())

            proj_layers.append(nn.Dropout(0.1))

            current_dim = target_dim

        self.proj_shared = nn.Sequential(*proj_layers)




        self.descriptor_audio = nn.Parameter(torch.randn(1, 1, descriptor_dim) * 0.02)

        self.descriptor_text = nn.Parameter(torch.randn(1, 1, descriptor_dim) * 0.02)

        self.descriptor_video = nn.Parameter(torch.randn(1, 1, descriptor_dim) * 0.02)




        desc_proj_layers = [

            nn.Linear(descriptor_dim, target_dim),

            nn.LayerNorm(target_dim),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(target_dim, target_dim),

            nn.LayerNorm(target_dim),

            nn.ReLU(),

        ]

        self.proj_descriptor = nn.Sequential(*desc_proj_layers)




        f_spec_layers = []

        f_spec_input_dim = target_dim * 3




        for i in range(n_layers):

            if i == 0:

                current_input_dim = f_spec_input_dim

                next_dim = target_dim * 2

            elif i == n_layers - 1:

                next_dim = target_dim

            else:

                current_input_dim = target_dim * 2

                next_dim = target_dim * 2


            f_spec_layers.append(nn.Linear(current_input_dim, next_dim))


            if i < n_layers - 1:

                f_spec_layers.append(nn.LayerNorm(next_dim))

                f_spec_layers.append(nn.ReLU())

                f_spec_layers.append(nn.Dropout(0.1))


        self.f_spec = nn.Sequential(*f_spec_layers)




        up_layers = [

            nn.Linear(target_dim, target_dim * 2),

            nn.LayerNorm(target_dim * 2),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(target_dim * 2, target_dim * 2),

            nn.LayerNorm(target_dim * 2),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(target_dim * 2, target_dim),

        ]

        self.upsampler = nn.Sequential(*up_layers)


    def forward(self, s_v, s_t, target_oracle=None, missing_modality=""):

        ""

        batch_size = s_v.size(0)

        seq_len = min(s_v.size(1), s_t.size(1))




        s_v_proj = self.proj_shared(s_v)[:, :seq_len, :]



        s_t_proj = self.proj_shared(s_t)[:, :seq_len, :]






        if missing_modality =="":

            descriptor = self.descriptor_audio

        elif missing_modality =="":

            descriptor = self.descriptor_text

        else:



            descriptor = self.descriptor_video


        descriptor_proj = self.proj_descriptor(descriptor)



        descriptor_expanded = descriptor_proj.expand(batch_size, seq_len, -1)




        combined = torch.cat([descriptor_expanded, s_v_proj, s_t_proj], dim=-1)

        reconstructed = self.f_spec(combined)




        loss_spec = torch.tensor(0.0, device=s_v.device)


        if target_oracle is not None:



            target_len = target_oracle.size(1)

            if target_len != seq_len:



                reconstructed = reconstructed.permute(0, 2, 1)



                reconstructed = F.interpolate(reconstructed, size=target_len, mode="", align_corners=False)

                reconstructed = reconstructed.permute(0, 2, 1)



                reconstructed = self.upsampler(reconstructed)




            loss_spec = F.mse_loss(reconstructed, target_oracle)


        return reconstructed, loss_spec



class Autoencoder(nn.Module):

    """
    RFR 基础自编码器单元。编码: input_size -> hidden_size，解码: hidden_size -> hidden_size。
    用于堆叠成 IF_IM 迭代想象链。
    """

    def __init__(self, input_size, hidden_size):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.decoder = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=0.1)

        self.activation = nn.ReLU()


    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.dropout(self.activation(self.decoder(encoded)))

        return decoded



class IF_IM(nn.Module):

    """
    RFR IF_IM 迭代想象链: 堆叠自编码器 + concat prompt + linear 映射。
    保持与原版相同的接口: (h) -> (B, L, shared_dim)。
    """

    def __init__(self, input_size, shared_dim=256, n_layers=3, prompt_dim=None):

        super(IF_IM, self).__init__()

        self.shared_dim = shared_dim

        if prompt_dim is None:

            prompt_dim = shared_dim

        self.prompt_dim = prompt_dim


        ae_input_size = input_size + prompt_dim

        self.autoencoders = nn.ModuleList(

            [Autoencoder(ae_input_size, shared_dim)] +

            [Autoencoder(shared_dim, shared_dim) for _ in range(n_layers - 1)]

        )

        self.prompt = nn.Parameter(torch.randn(1, prompt_dim) * 0.02, requires_grad=True)


        self.linear = nn.Sequential(

            nn.Linear(shared_dim, shared_dim),

            nn.ReLU(),

            nn.Dropout(0.1)

        )


    def forward(self, h):



        original_shape = h.shape



        is_3d = len(original_shape) == 3

        L = original_shape[1] if is_3d else 1


        if is_3d:



            h = h.reshape(-1, h.size(-1))




        prompt_expanded = self.prompt.repeat(h.shape[0], 1)




        delta_z_list = []

        delta_z = self.autoencoders[0](torch.cat((h, prompt_expanded), dim=1))

        delta_z_list.append(delta_z)


        for i in range(1, len(self.autoencoders)):

            delta_z = self.autoencoders[i](delta_z_list[i - 1])

            delta_z_list.append(delta_z)


        h_imagined = self.linear(delta_z_list[-1])


        if is_3d:



            h_imagined = h_imagined.reshape(original_shape[0], L, -1)



            if h_imagined.size(1) > 1:

                diff = h_imagined[0, 1:] - h_imagined[0, :1]




        return h_imagined



class CMDLoss(nn.Module):

    """
    Central Moment Discrepancy Loss — 对齐两个模态的分布。
    与 RFR 原版 CMD 实现保持一致。
    """

    def __init__(self):

        super(CMDLoss, self).__init__()


    def matchnorm(self, x1, x2):

        power = torch.pow(x1 - x2, 2)

        summed = torch.sum(power)

        return summed ** 0.5


    def scm(self, sx1, sx2, k):

        ss1 = torch.mean(torch.pow(sx1, k), dim=0)

        ss2 = torch.mean(torch.pow(sx2, k), dim=0)

        return self.matchnorm(ss1, ss2)


    def forward(self, x1, x2, n_moments=2):

        x1_flat = x1.view(-1, x1.size(-1))

        x2_flat = x2.view(-1, x2.size(-1))

        mx1 = torch.mean(x1_flat, dim=0)

        mx2 = torch.mean(x2_flat, dim=0)

        sx1 = x1_flat - mx1

        sx2 = x2_flat - mx2

        dm = self.matchnorm(mx1, mx2)

        scms = dm

        for i in range(n_moments - 1):

            scms += self.scm(sx1, sx2, i + 2)

        return scms



class LossFunctions(nn.Module):

    """
    RFR 损失函数集合: shared_loss, spec_loss。
    与原版 LossFunctions 保持一致。
    """

    def __init__(self):

        super(LossFunctions, self).__init__()

        self.loss_cmd_func = CMDLoss()


    def shared_loss(self, invariance_x, invariance_y):

        loss = self.loss_cmd_func(invariance_x, invariance_y, 2)

        return loss


    def spec_loss(self, H_predicted, H_target):



        mse_loss = torch.mean((H_predicted - H_target) ** 2)


        if H_predicted.size(1) > 1:

            diff_pred = H_predicted[:, 1:] - H_predicted[:, :-1]

            diff_target = H_target[:, 1:] - H_target[:, :-1]

            temporal_diff_loss = torch.mean((diff_pred - diff_target) ** 2)

        else:

            temporal_diff_loss = torch.tensor(0.0, device=H_predicted.device)


        return mse_loss + temporal_diff_loss



class ModalSpecificReconstructor(nn.Module):

    """
    RFR 模态特定重建器: 拼接特征描述符 + 池化特征，重建目标模态。
    保持与原版相同的接口: (s_v, s_t, oracle) -> (reconstructed, loss_spec)。
    """

    def __init__(self, shared_dim=256, target_dim=512, max_seq_len=5000):

        super(ModalSpecificReconstructor, self).__init__()

        self.target_dim = target_dim

        self.max_seq_len = max_seq_len




        self.proj_v = nn.Sequential(

            nn.Linear(shared_dim, target_dim),

            nn.LayerNorm(target_dim),

            nn.ReLU(),

            nn.Dropout(0.1)

        )

        self.proj_t = nn.Sequential(

            nn.Linear(shared_dim, target_dim),

            nn.LayerNorm(target_dim),

            nn.ReLU(),

            nn.Dropout(0.1)

        )


        self.descriptor_embed = nn.Parameter(torch.randn(1, 1, target_dim) * 0.02)

        self.layer_norm = nn.LayerNorm(target_dim * 3)


        self.f_spec = nn.Sequential(

            nn.Linear(target_dim * 3, target_dim),

            nn.LayerNorm(target_dim),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(target_dim, target_dim),

        )




        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, target_dim) * 0.02)




        self.temporal_decoder = nn.Sequential(

            nn.Linear(target_dim, target_dim),

            nn.ReLU(),

            nn.Linear(target_dim, target_dim),

        )


        self.prompt_recon = nn.Parameter(torch.randn(1, 1, target_dim) * 0.02, requires_grad=True)

        self.recon_mlp = nn.Sequential(

            nn.Linear(target_dim, target_dim),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(target_dim, target_dim),

        )




        self.upsampler = nn.Sequential(

            nn.Linear(target_dim, target_dim * 4),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(target_dim * 4, target_dim * 4),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(target_dim * 4, target_dim),

        )


        self.loss_functions = LossFunctions()


    def forward(self, s_v, s_t, oracle=None):

        batch_size = s_v.size(0)




        s_v_proj = self.proj_v(s_v)



        s_t_proj = self.proj_t(s_t)






        seq_len = min(s_v_proj.size(1), s_t_proj.size(1))

        s_v_proj = s_v_proj[:, :seq_len, :]

        s_t_proj = s_t_proj[:, :seq_len, :]




        descriptor_expanded = self.descriptor_embed.expand(batch_size, seq_len, -1)




        combined_features = torch.cat([descriptor_expanded, s_v_proj, s_t_proj], dim=-1)

        combined_features = self.layer_norm(combined_features)


        reconstructed_feature = self.f_spec(combined_features)




        prompt_expanded = self.prompt_recon.expand(-1, reconstructed_feature.size(1), -1)

        reconstructed_feature = reconstructed_feature + self.recon_mlp(prompt_expanded)


        if oracle is not None:



            oracle_len = oracle.size(1)

            if oracle_len != seq_len:



                b, _, d = reconstructed_feature.shape

                reconstructed_feature = self._mlp_upsample(reconstructed_feature, oracle_len)

            reconstructed_feature = self.temporal_decoder(reconstructed_feature)


            loss_spec = self.loss_functions.spec_loss(reconstructed_feature, oracle)

        else:

            reconstructed_feature = self.temporal_decoder(reconstructed_feature)

            loss_spec = torch.tensor(0.0, device=reconstructed_feature.device)


        return reconstructed_feature, loss_spec


    def _mlp_upsample(self, x, target_len):

        ""

        b, seq_len, d = x.shape




        pos_embed = self.pos_embed[:, :max(seq_len, target_len), :]




        x = x.permute(0, 2, 1)



        x = x.unsqueeze(-1)



        x = x.squeeze(-1)






        upsampled = torch.nn.functional.interpolate(

            x, size=target_len, mode="", align_corners=False

        )






        upsampled = upsampled.permute(0, 2, 1)



        upsampled = self.upsampler(upsampled)




        return upsampled



class FeatureReconstructionNetwork(nn.Module):

    """
    CRN 风格的特征重建网络（深层版本）。
    - Shared Feature Learning: 使用深层 MLP 提取共享语义
    - Modality-Specific Learning: 使用深层 MLP 注入模态特定信息
    - CMD Loss: 对齐不同模态的共享表示分布
    """

    def __init__(self, dim_v, dim_t, dim_a_missing, n_layers=5):

        super(FeatureReconstructionNetwork, self).__init__()

        input_dim = dim_v + dim_t



        shared_dim = 256

        descriptor_dim = 128






        self.fsha_v = CRNSharedFeatureExtractor(

            input_dim, hidden_dim=512, output_dim=shared_dim, n_layers=n_layers

        )

        self.fsha_t = CRNSharedFeatureExtractor(

            input_dim, hidden_dim=512, output_dim=shared_dim, n_layers=n_layers

        )




        self.reconstructor = CRNModalityAwareReconstructor(

            shared_dim=shared_dim,

            target_dim=dim_a_missing,

            descriptor_dim=descriptor_dim,

            n_layers=n_layers

        )


        self.loss_cmd_func = CMDLoss()


    def forward(self, v, t, a_oracle=None, missing_modality=""):

        ""



        seq_len = min(v.size(1), t.size(1))

        v_aligned = v[:, :seq_len, :]

        t_aligned = t[:, :seq_len, :]




        s_v = self.fsha_v(v_aligned, t_aligned)



        s_t = self.fsha_t(t_aligned, v_aligned)






        loss_sha = self.loss_cmd_func(s_v, s_t, n_moments=2)




        reconstructed, loss_spec = self.reconstructor(

            s_v, s_t, a_oracle, missing_modality=missing_modality

        )


        loss_recon = loss_sha + loss_spec


        return reconstructed, loss_recon

