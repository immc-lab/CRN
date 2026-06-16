utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CRNSharedFeatureExtractor(nn.Module):
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
        combined = torch.cat(modalities, dim=-1)
        return self.net(combined)
class CRNModalityAwareReconstructor(nn.Module):
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
            currnt_dim = target_dim
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
            nn.ReLU()
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

class CMDLoss(nn.Module):
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
        self.pompt_recon = nn.Parameter(torch.randn(1, 1, target_dim) * 0.02, requires_grad=True)
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
        seq_len = min(v.size(1), t.size(1))
        v_aligned = v[:, :seq_len, :]
        t_aligned = t[:, :seq_len, :]
        s_v = self.fsha_v(v_aligned, t_aligned)
        s_t = self.fsha_t(t_aligned, v_aligned)
        loss_sha = self.loss_cmd_func(s_v, s_t, n_moments=None)
        reconstructed, loss_spec = self.reconstructor(
            s_v, s_t, a_oracle, missing_modality=missing_modality
        )
        loss_recon = loss_sha + loss_spec
        return reconstructed, loss_recon

