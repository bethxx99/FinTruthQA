import torch
from torch import nn
from typing import Optional
from transformers import AutoConfig, AutoModel


class BertClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**inputs)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))
