import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.rank = args.rank
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.dropepoch=args.dropepoch
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
    def forward(self, text_features, image_features,local,mlm_scores,mlm_labels,epoch,threshold , logit_scale=2.659):

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        device = image_features.device
        sims = image_features @ text_features.T
        if epoch<self.dropepoch:
            logits_per_image = logit_scale * sims
            logits_per_text = logit_scale * sims.T
            local1 = local
            local2 = local.T
        else:
            true = torch.diag(sims)
            reducelabel = true >threshold
            resize_image = sims[reducelabel,:]
            resize_text = sims.T[reducelabel,:]
            logits_per_image = logit_scale * resize_image
            logits_per_text = logit_scale * resize_text 
            local1 = local[reducelabel,:]
            local2 = local.T[reducelabel,:]

        # calculated ground-truth and cache if enabled
        # num_logits = logits_per_image.shape[0]  
        num_logits = sims.shape[0]  
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]

        if epoch <self.dropepoch:
            resize_label = labels
        else:
            resize_label = labels[reducelabel]

        # total_loss = (
        #     F.cross_entropy(logits_per_image, labels) +
        #     F.cross_entropy(logits_per_text, labels)
        #     ) 
        total_loss = (
            F.cross_entropy(logits_per_image, resize_label) +
            F.cross_entropy(logits_per_text, resize_label)
            ) 
        localloss2 = F.cross_entropy(local1, resize_label) + F.cross_entropy(local2, resize_label)

        mlmloss=self.ce(mlm_scores, mlm_labels)
        return total_loss+localloss2+0.5*mlmloss
