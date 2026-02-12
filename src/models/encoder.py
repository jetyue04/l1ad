import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, intermediate_architecture, bottleneck_size, drop_out=None):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, intermediate_architecture[0]))
        for n_neurons_last, n_neurons in zip(intermediate_architecture[:-1],
                                             intermediate_architecture[1:]):
            self.layers.append(nn.Linear(n_neurons_last, n_neurons))
        self.layers.append(nn.Linear(intermediate_architecture[-1], bottleneck_size))
        self.n_layers = len(self.layers)

        if drop_out is not None:
            if isinstance(drop_out, (float, int)):
                drop_out = [drop_out] * self.n_layers
            if len(drop_out) != self.n_layers:
                print("ERROR: Drop out array must have the same size as number of layers")
                exit(1)
            for i_layer in range(self.n_layers):
                drop_out_coef = drop_out[i_layer]
                self.dropout_layers.append(nn.Dropout(drop_out_coef))

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < self.n_layers-1:
                x = nn.functional.relu(x)
            if len(self.dropout_layers) > idx:
                x = self.dropout_layers[idx](x)
        return x


