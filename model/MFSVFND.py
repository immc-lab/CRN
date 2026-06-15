import torch
import torch.nn as nn

from model.cross_attention_module import CrossTransformer
from model.modality_attention_module import ModalityTransformer
from model.script import CMDLoss, CRNSharedFeatureExtractor, CRNModalityAwareReconstructor, FeatureReconstructionNetwork
from utils.tools import *


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(2)).view(b, c)
        y = self.fc(y)
        return x * y


class classifier(nn.Module):
    def __init__(self, fea_dim, dropout_probability):
        super(classifier, self).__init__()
        self.class_net = nn.Sequential(
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(32, 2)
        )

    def forward(self, fea):
        out = self.class_net(fea)
        return out


class MFSVFNDModel(torch.nn.Module):
    def __init__(self, fea_dim, dropout, dataset, missing_modality='none'):
        super(MFSVFNDModel, self).__init__()
        VALID = {'none', 't', 'v', 'a', 'tv', 'ta', 'va'}
        if missing_modality not in VALID:
            raise ValueError(f'missing_modality must be one of {VALID}, got {missing_modality}')
        self.missing_modality = missing_modality
        if dataset == 'fakesv':
            self.bert = pretrain_bert_wwm_model()
            self.text_dim = 1024
        else:
            self.bert = pretrain_bert_uncased_model()
            self.text_dim = 768

        self.img_dim = 1024
        self.audio_dim = 1024

        self.dim = fea_dim
        self.num_heads = 8
        self.trans_dim = 512
        self.dropout = dropout

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),
                                         nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.trans_dim), torch.nn.ReLU(),
                                        nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(self.audio_dim, self.trans_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))

        # text
        self.text_causal_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)
        self.text_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)

        # image
        self.image_causal_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)
        self.image_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)

        # audio
        self.audio_causal_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)
        self.audio_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)


        # CrossTransformer
        self.tv_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.ta_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.av_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.vt_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.at_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.va_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)


        # Learnable residual scale for unimodal shortcut.
        self.residual_scale = nn.Parameter(torch.tensor(0.8))
        self.se_module = SE(channels=512, reduction=16)
        self.classifier = classifier(fea_dim=512, dropout_probability=self.dropout)

        # Reconstruction networks: one per missing-modality scenario.
        # All modalities share trans_dim=512.
        self.recon_t = FeatureReconstructionNetwork(self.trans_dim, self.trans_dim, self.trans_dim)
        self.recon_v = FeatureReconstructionNetwork(self.trans_dim, self.trans_dim, self.trans_dim)
        self.recon_a = FeatureReconstructionNetwork(self.trans_dim, self.trans_dim, self.trans_dim)

        self.cmd_loss_fn = CMDLoss()

    def forward(self, **kwargs):
        ### Title ###
        title_inputid = kwargs['title_inputid']  # (batch,512,D)
        title_mask = kwargs['title_mask']
        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']
        fea_text = self.linear_text(fea_text)

        ### Audio Frames ###
        fea_audio = kwargs['audio_feas']  # (B,L,D)
        fea_audio = self.linear_audio(fea_audio)

        ### Image Frames ###
        frames = kwargs['frames']  # (B,L,D)
        fea_image = self.linear_img(frames)

        # 保存用于重建 oracle 的原始特征（在 linear_* 变换之后、transformer 之前）
        # linear_* 的输出维度为 trans_dim=512，与 ModalSpecificReconstructor 的 target_dim 一致
        text_oracle = fea_text.clone()
        image_oracle = fea_image.clone()
        audio_oracle = fea_audio.clone()

        # text
        fea_text = self.text_causal_transformer(fea_text)
        fea_text = self.text_transformer(fea_text)

        # image
        fea_image = self.image_causal_transformer(fea_image)
        fea_image = self.image_transformer(fea_image)

        # audio
        fea_audio = self.audio_causal_transformer(fea_audio)
        fea_audio = self.audio_transformer(fea_audio)

        # Reconstruct missing modality using the FeatureReconstructionNetwork.
        m = self.missing_modality
        loss_recon = torch.tensor(0.0, device=fea_text.device)

        if m == 't':
            fea_text, lr = self.recon_t(fea_image, fea_audio, text_oracle, missing_modality='t')
           # fea_text = torch.clamp(fea_text, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr
        elif m == 'v':
            fea_image, lr = self.recon_v(fea_text, fea_audio, image_oracle, missing_modality='v')
            #fea_image = torch.clamp(fea_image, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr
        elif m == 'a':
            fea_audio, lr = self.recon_a(fea_text, fea_image, audio_oracle, missing_modality='a')
            #fea_audio = torch.clamp(fea_audio, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr
        elif m == 'tv':
            # Missing text+video, only audio available
            fea_text, lr_t = self.recon_t(fea_audio, fea_audio, text_oracle, missing_modality='t')
            fea_text = torch.clamp(fea_text, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr_t

            fea_image, lr_v = self.recon_v(fea_audio, fea_audio, image_oracle, missing_modality='v')
            fea_image = torch.clamp(fea_image, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr_v
        elif m == 'ta':
            # Missing text+audio, only video available
            fea_text, lr_t = self.recon_t(fea_image, fea_image, text_oracle, missing_modality='t')
            fea_text = torch.clamp(fea_text, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr_t

            fea_audio, lr_a = self.recon_a(fea_image, fea_image, audio_oracle, missing_modality='a')
            fea_audio = torch.clamp(fea_audio, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr_a
        elif m == 'va':
            # Missing video+audio, only text available
            fea_image, lr_v = self.recon_v(fea_text, fea_text, image_oracle, missing_modality='v')
            fea_image = torch.clamp(fea_image, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr_v

            fea_audio, lr_a = self.recon_a(fea_text, fea_text, audio_oracle, missing_modality='a')
            fea_audio = torch.clamp(fea_audio, min=-30.0, max=30.0)
            loss_recon = loss_recon + lr_a

        # cross_attention
        fea_tv = self.tv_cross_transformer(fea_text, fea_image)
        fea_vt = self.vt_cross_transformer(fea_image, fea_text)

        fea_ta = self.ta_cross_transformer(fea_text, fea_audio)
        fea_at = self.at_cross_transformer(fea_audio, fea_text)

        fea_va = self.va_cross_transformer(fea_image, fea_audio)
        fea_av = self.av_cross_transformer(fea_audio, fea_image)

        # Pool bidirectional cross-modal features to fixed-size vectors.
        fea_tv = torch.mean(torch.cat((fea_tv, fea_vt), dim=1), dim=1)
        fea_ta = torch.mean(torch.cat((fea_ta, fea_at), dim=1), dim=1)
        fea_va = torch.mean(torch.cat((fea_va, fea_av), dim=1), dim=1)

        # Unimodal residual shortcut (global average pooling per modality).
        fea_text_res = torch.mean(fea_text, dim=1)
        fea_image_res = torch.mean(fea_image, dim=1)
        fea_audio_res = torch.mean(fea_audio, dim=1)

        final_fea =fea_ta+fea_tv+fea_va+self.residual_scale * (fea_text_res + fea_image_res + fea_audio_res)

        output = self.classifier(final_fea)

        return output, loss_recon
