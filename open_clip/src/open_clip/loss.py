import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def gather_embeddings( #i.e. gather tokens
        image_embeddings,
        text_embeddings,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    """Used to gather the embeddings from each from the entire world"""
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        raise NotImplementedError
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_embeddings = torch.cat(torch.distributed.nn.all_gather(image_embeddings), dim=0)
            all_text_embeddings = torch.cat(torch.distributed.nn.all_gather(text_embeddings), dim=0)
        else:
            gathered_image_embeddings = [torch.zeros_like(image_embeddings) for _ in range(world_size)]
            gathered_text_embeddings = [torch.zeros_like(text_embeddings) for _ in range(world_size)]
            dist.all_gather(gathered_image_embeddings , image_embeddings)
            dist.all_gather(gathered_text_embeddings , text_embeddings)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_embeddings[rank] = image_embeddings 
                gathered_text_embeddings[rank] = text_embeddings 
            all_image_embeddings = torch.cat(gathered_image_embeddings, dim=0)
            all_text_embeddings = torch.cat(gathered_text_embeddings, dim=0)

    return all_image_embeddings, all_text_embeddings 


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class ColbertLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            similarity_metric='cosine',
            dropout=0.,
            global_contrastive='all',
            local_contrastive='all',
            pairwise_loss=False,
            clip_context_length=None
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.similarity_metric = similarity_metric

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        # loss information
        if dropout != 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.global_contrastive=global_contrastive
        self.local_contrastive=local_contrastive
        self.pairwise_loss = pairwise_loss

    def get_embeddings(self, image_embeddings, text_embeddings):
        if self.world_size > 1:
            if self.local_loss:
                logits_per_image, logits_per_text = image_embeddings, text_embeddings
            else:
                all_image_embeddings, all_text_embeddings = gather_embeddings(
                        image_embeddings, text_embeddings,
                        self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
                logits_per_image, logits_per_text = all_image_embeddings, all_text_embeddings
        else:
            logits_per_image, logits_per_text = image_embeddings, text_embeddings
        return logits_per_image, logits_per_text
        
    def check_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            logging.info(f"NaN detected in {name}")

    def get_pairwise_mask(self, c, i, device='cpu'):
        block_size = 2
        mask = torch.full((c, i), float('-inf'), device=device, requires_grad=False)
        
        num_blocks = min(c, i) // block_size
        
        block = torch.zeros((block_size, block_size), device=device)
        
        for idx in range(num_blocks):
            start_idx = idx * block_size
            mask[start_idx:start_idx + block_size, start_idx:start_idx + block_size] = block
        
        return mask
    
    def forward(self, image_features, text_features, image_embeddings, text_embeddings, logit_scale, output_dict=False, masks=None):
        if masks is not None:
            text_embeddings = text_embeddings * masks.unsqueeze(-1)
        
        similarity = torch.einsum('ctd,ipd->citp', text_embeddings, image_embeddings) * logit_scale
        
        c, i, _, _ = similarity.shape
        similarity_mask = self.get_pairwise_mask(c, i, device=similarity.device) if self.pairwise_loss else None

        if self.dropout is not None:
            similarity = self.dropout(similarity)

        # Apply MaxSim operator - maximum similarity across all documents for each query term, and vice versa
        scores = []
        if self.local_contrastive == "patch-wise" or self.local_contrastive == "all":
            # Take the maxsim for every patch
            if masks is not None:
                max_sim_p = similarity.amax(dim=3).mean(dim=-1)
            else:
                max_sim_p = similarity.amax(dim=2).sum(dim=-1)
            scores.append(max_sim_p)

        if self.local_contrastive == "token-wise" or self.local_contrastive == "all":
            # Take the maxsim for every token
            if masks is not None:
                max_sim_t = similarity.amax(dim=2).mean(dim=-1)
            else:
                max_sim_t = similarity.amax(dim=3).sum(dim=-1)
            scores.append(max_sim_t)

        # Apply Constrastive Loss
        loss_streams = []
        if self.global_contrastive == "image-wise" or self.global_contrastive == "all":
            for score in scores:
                if similarity_mask is not None:
                    score = score + similarity_mask
                image_wise_softmax = torch.softmax(score, dim=0)
                image_wise_loss = -torch.log(image_wise_softmax.diag() + 1e-8).mean()
                loss_streams.append(image_wise_loss)

        if self.global_contrastive == "text-wise" or self.global_contrastive == "all":
            for score in scores:
                if similarity_mask is not None:
                    score = score + similarity_mask
                text_wise_softmax = torch.softmax(score, dim=1)
                text_wise_loss = -torch.log(text_wise_softmax.diag() + 1e-8).mean()
                loss_streams.append(text_wise_loss)

        # Combine the losses for the final contrastive loss
        contrastive_loss = torch.stack(loss_streams).mean()
        return {"contrastive_loss": contrastive_loss} if output_dict else contrastive_loss

class SparcLoss(nn.Module):
    """# This requires that we have access to the tokens, this has not been implemented on
    CustomTextTowerClip, so it will throw an error."""

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            local_lambda=1.0,
            global_lambda=1.0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        # Set Parameters
        self.global_lambda = global_lambda
        self.local_lambda = local_lambda

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, image_embeddings, text_embeddings, logit_scale, output_dict=False):
        # ---------- Global Loss ------------

        #Get similarity matrix
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        global_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        # ---------- Local Loss ------------
        # similarity calculation
        similarity = torch.einsum('btd,bpd->btp', text_embeddings, image_embeddings)

        # min-max normalization
        similarity = (similarity - torch.amin(similarity, dim=-1, keepdim=True)) /         \
                     (torch.amax(similarity, dim=-1, keepdim=True) - torch.amin(similarity, dim=-1, keepdim=True))

        # thresholding
        similarity = torch.where(similarity < 1 / similarity.shape[-1], 0.0, similarity)

        # alignment-weighting
        image_align_weights = similarity / torch.sum(similarity, dim=-1, keepdim=True)
        text_grouped_image_patch_embed = torch.einsum('btp,bpd->btd', image_align_weights, image_embeddings)

        # Normalize Embeddings 
        local_image_tokens = text_grouped_image_patch_embed / text_grouped_image_patch_embed.norm(dim=-1, keepdim=True)
        local_text_tokens = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # Take pairwise contrastive loss
        # IDK how to do this with pytorch CrossEntropyLoss as it is nested cross-entropy
        # This can be fixed we are doing to many extras ops
        cross_product = torch.einsum('btd,bpd->btp', local_text_tokens, local_image_tokens) * logit_scale
        logits_image_token = F.softmax(cross_product, dim=2) # Take column-wise softmax
        logits_text_token = F.softmax(cross_product, dim=1) # Take row-wise softmax

        # Take Log and reduce 
        pairwise_loss = torch.log(logits_text_token) + torch.log(logits_image_token)
        local_loss = (-1 / pairwise_loss.shape[0]) * torch.sum(torch.einsum('btt->b', pairwise_loss) / cross_product.shape[1])

        # ---------- Total Loss ------------
        total_loss = self.global_lambda * global_loss + self.local_lambda * local_loss
        return {"contrastive_loss": total_loss} if output_dict else total_loss

class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss

class ColbertDistillClipLoss(ColbertLoss):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            similarity_metric='cosine',
            dropout=0.,
            global_contrastive='all',
            local_contrastive='all',
            include_contrastive = False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
            similarity_metric=similarity_metric,
            dropout=dropout,
            global_contrastive=global_contrastive,
            local_contrastive=local_contrastive
        )
        self.include_contrastive = include_contrastive

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    

    def forward(
        self,
        image_features,
        text_features,
        image_embeddings,
        text_embeddings,
        dist_image_features,
        dist_text_features,
        dist_logit_scale,
        logit_scale,
        output_dict=False
        ):

        image_embeddings, text_embeddings = self.get_embeddings(image_embeddings, text_embeddings)
        dist_image_features, dist_text_features = self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        # Reapplied logit_scale (We might need to retest with this)
        similarity = torch.einsum('ctd,ipd->citp', text_embeddings, image_embeddings) * logit_scale # Change is here

        # Distill Similarity
        distill_similarity = torch.einsum("cd,id->ci", dist_text_features, dist_image_features)

        if self.dropout is not None:
            similarity = self.dropout(similarity)

        # Apply MaxSim operator - maximum similarity across all documents for each query term, and vice versa
        scores = []
        if self.local_contrastive == "patch-wise" or self.local_contrastive == "all":
            # Take the maxsim for every patch
            max_sim_p = similarity.amax(dim=2).sum(dim=-1)
            scores.append(max_sim_p)

        if self.local_contrastive == "token-wise" or self.local_contrastive == "all":
            # Take the maxsim for every token
            max_sim_t = similarity.amax(dim=3).sum(dim=-1)
            scores.append(max_sim_t)

        # Apply Constrastive Loss
        loss_streams = []
        if self.global_contrastive == "image-wise" or self.global_contrastive == "all":
            for score in scores:
                image_wise_softmax = torch.softmax(score, dim=0)
                image_wise_loss = -torch.log(image_wise_softmax.diag() + 1e-8).mean()
                loss_streams.append(image_wise_loss)

        if self.global_contrastive == "text-wise" or self.global_contrastive == "all":
            for score in scores:
                text_wise_softmax = torch.softmax(score, dim=1)
                text_wise_loss = -torch.log(text_wise_softmax.diag() + 1e-8).mean()
                loss_streams.append(text_wise_loss)

        # Apply Distillation Loss
        # Should these be means or sums
        distill_loss_streams = []
        if self.global_contrastive == "image-wise" or self.global_contrastive == "all":
            for score in scores:
                image_wise_softmax = torch.softmax(score, dim=0)
                dist_image_wise_softmax = torch.softmax(distill_similarity, dim=0)
                image_wise_nll = dist_image_wise_softmax * (-torch.log(image_wise_softmax + 1e-8))
                image_wise_loss = image_wise_nll.sum(dim=0).mean(dim=0)
                distill_loss_streams.append(image_wise_loss)

        # Add log softmax
        if self.global_contrastive == "text-wise" or self.global_contrastive == "all":
            for score in scores:
                text_wise_softmax = torch.softmax(score, dim=1)
                dist_text_wise_softmax = torch.softmax(distill_similarity, dim=1)
                text_wise_nll = dist_text_wise_softmax * (-torch.log(text_wise_softmax + 1e-8))
                text_wise_loss = text_wise_nll.sum(dim=1).mean(dim=0)
                distill_loss_streams.append(text_wise_loss)

        # Combine the losses for the final contrastive loss
        contrastive_loss = torch.stack(loss_streams).mean()
        distill_loss = torch.stack(distill_loss_streams).mean()
        if self.include_contrastive:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss} if output_dict else (contrastive_loss, distill_loss)
        else:
            return {"distill_loss": distill_loss} if output_dict else distill_loss

class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
