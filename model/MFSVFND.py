import torch
import torch.nn as nn

from model.script import CMDLoss, CRNSharedFeatureExtractor, CRNModalityAwareReconstructor, FeatureReconstructionNetwork
from utils.tools import *




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
        self.recon_t = FeatureReconstructionNetwork(self.trans_dim, self.trans_dim, self.trans_dim)
        self.recon_v = FeatureReconstructionNetwork(self.trans_dim, self.trans_dim, self.trans_dim)
        self.recon_a = FeatureReconstructionNetwork(self.trans_dim, self.trans_dim, self.trans_dim)

        self.cmd_loss_fn = CMDLoss()

    def forward(self, **kwargs):
        title_inputid = kwargs['title_inputid']  # (batch,512,D)
        title_mask = kwargs['title_mask']
        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']
        fea_text = self.linear_text(fea_text)
        fea_audio = kwargs['audio_feas']  # (B,L,D)
        fea_audio = self.linear_audio(fea_audio)

        frames = kwargs['frames']  # (B,L,D)
        fea_image = self.linear_img(frames)

        text_oracle = fea_text.clone()
        image_oracle = fea_image.clone()
        audio_oracle = fea_audio.clone()

    
        fea_text = self.text_causal_transformer(fea_text)
        fea_text = self.text_transformer(fea_text)

        fea_image = self.image_causal_transformer(fea_image)
        fea_image = self.image_transformer(fea_image)

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

        
        fea_text = torch.mean(fea_text, dim=1)
        fea_image = torch.mean(fea_image, dim=1)
        fea_audio = torch.mean(fea_audio, dim=1)

        final_fea=fea_text + fea_image + fea_audio)

        output = self.classifier(final_fea)

        return output, loss_recon
