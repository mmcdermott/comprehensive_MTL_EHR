import torch.nn as nn

class MultitaskHead(nn.Module):
    def __init__(self, config, task_dims):
        super().__init__()
        self.task_layers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, task_dim) for task, task_dim in task_dims.items()
        })

    def forward(self, pooled_output):
        return {task: layer(pooled_output) for task, layer in self.task_layers.items()}
