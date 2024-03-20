import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class XLMRobertaForNER(nn.Module):
    def __init__(self, model_name, num_labels, freeze_xlmr=True):
        super(XLMRobertaForNER, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(model_name)

        self.classifier = nn.Linear(self.xlmr.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None):
        outputs = self.xlmr(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits
