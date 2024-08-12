import torch
from torch import nn

from distilrl.pl_base import LightningBase
from distilrl.dt import TransformerBlock
from typing import Optional, Literal

from torchmetrics import Accuracy

class CausalBase(LightningBase):
    """Lightning class for classification task"""
    
    def step(self, batch, batch_idx):
        
        actions, mask = batch["actions"], batch["mask"]
        padding_mask = ~mask.to(torch.bool)
        predicted_actions = self(
            *batch[-1],
            padding_mask=padding_mask,
        )
        
        loss = self.loss(predicted_actions, actions.detach(), reduction="none")
        loss = (loss * mask.unsqueeze(-1)).mean()
        
        return {"loss": loss}


class DecisionTransformer(CausalBase):
    def __init__(
        self,
        num_actions: int,
        num_states: int | None = None,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
        emb_strategy: str = "stack",
        feedforward_dim: int | None = 512,
        optimizer = None,
        scheduler = None,
        optimizer_kwargs = None,
        scheduler_kwargs = None,
        loss = nn.CrossEntropyLoss,
    ):
        super().__init__(loss=loss, optimizer=optimizer, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, optimizer_kwargs=optimizer_kwargs)
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)
        self.emb_strategy = emb_strategy
        
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        if num_states is not None:
            self.state_emb = nn.Embedding(num_states, embedding_dim)
        self.action_emb = nn.Embedding(num_actions, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.seq_len = seq_len
        self.num_stacks = 1 + int(self.emb_strategy == "stack") + int(num_states is not None and num_states != 0)
        self.seq_len *= self.num_stacks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=self.seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    feedforward_dim=feedforward_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Linear(embedding_dim, num_actions)
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.num_states = num_states
        self.num_actions = num_actions
        self.episode_len = episode_len
        self.max_action = max_action
        
        self.conv_out = nn.Conv1d(self.seq_len, self.seq_len // self.num_stacks, 1)

    #     self.apply(self._init_weights)

    # @staticmethod
    # def _init_weights(module: nn.Module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.LayerNorm):
    #         torch.nn.init.zeros_(module.bias)
    #         torch.nn.init.ones_(module.weight)

    def forward(
        self,
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        rewards: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        states: Optional[torch.Tensor] = None,  # [batch_size, seq_len, state_dim]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size = actions.shape[0]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        act_emb = self.action_emb(actions) + time_emb
        rewards_emb = self.return_emb(rewards.unsqueeze(-1)) + time_emb
        state_emb = None
        if self.num_stacks == 3:
            state_emb = self.state_emb(states) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        emb_list = [x for x in [rewards_emb, state_emb, act_emb] if x is not None]
        sequence = torch.stack(emb_list, dim=1)
        if self.emb_strategy == "stack":
            sequence = (
                sequence
                .permute(0, 2, 1, 3)
                .reshape(batch_size, self.seq_len, self.embedding_dim)
            )
        else:
            sequence = sequence.sum(1).reshape(batch_size, self.seq_len, self.embedding_dim)
            
        padding_list = [padding_mask] * self.num_stacks
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack(padding_list, dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, self.seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.action_head(self.conv_out(self.out_norm(out)))
        return out[:, 0, :]
    
    def step(self, batch, batch_idx, subset="train", on_step=False, on_epoch=True,
            prog_bar=True, logger=True):   
        
        # remove squeeze, should not be here
        future_actions, future_rewards = batch["future_actions"].squeeze(1), batch["future_rewards"]
            
        args = {k: v for k, v in batch.items() if "future" not in k}
        predicted_actions = self(**args)
        
        loss = self.loss(predicted_actions, future_actions.detach()).mean()
        
        scorer = Accuracy("multiclass", num_classes=self.num_actions).to(self.device)
        accuracy = scorer(predicted_actions, future_actions)
        
        # нужно как-то поумнее вставить
        # mse = nn.MSELoss()
        # mean_regret = mse(predicted_actions)
        
        res = {"loss": loss, "accuracy": accuracy}
        print(res, accuracy)
        if subset != "predict":
            self.log(f"{subset}_loss", loss.detach().item(), on_step=on_step, on_epoch=on_epoch,
            prog_bar=prog_bar, logger=logger)
            self.log(f"{subset}_accuracy", accuracy.item(), on_step=on_step, on_epoch=on_epoch,
            prog_bar=prog_bar, logger=logger)
        else:
            res["pred"] = pred
            res["true"] = label
        return res