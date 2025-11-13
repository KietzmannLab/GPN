import torch
import torch.nn as nn
import torch.nn.functional as F
import math, contextlib
from torchvision.models import resnet18

class lstm_gpn(nn.Module):

    def __init__(self, timestep_multiplier=3, glimpse_loss=1, semantic_loss=0, scene_loss=0, gazeloc_loss=0, n_rnn=256, regularisation=1, input_dropout=0.25, rnn_dropout=0.1, return_all_actvs=0, input_split=0, recurrence=True, input_feats=2048):

        super(lstm_gpn, self).__init__()

        self.actv_dropout = nn.Dropout(input_dropout) if regularisation else nn.Identity()
        self.actv_proj = nn.Linear(input_feats, int(n_rnn/2))
        self.actv_proj_norm = nn.LayerNorm(int(n_rnn/2)) if regularisation else nn.Identity()
        self.coord_proj = nn.Linear(2, int(n_rnn/2))
        self.coord_proj_norm = nn.LayerNorm(int(n_rnn/2)) if regularisation else nn.Identity()
        self.return_all_actvs = return_all_actvs # this is for analysis
        self.input_split = input_split # 0 if provide image and saccade together and ask for prediction per timestep, 1 if provide image and saccade together and ask for prediction at odd timesteps (when saccade is provided)
        self.recurrence = recurrence # True if recurrence is enabled, False if not

        self.timestep_multiplier = timestep_multiplier
        self.lstm = nn.LSTM(n_rnn,n_rnn,self.timestep_multiplier,dropout = rnn_dropout if regularisation else 0,batch_first=True) if self.recurrence else NonRecurrentLSTM(n_rnn,n_rnn,self.timestep_multiplier,dropout = rnn_dropout if regularisation else 0,batch_first=True)
        self.lstm_norm = nn.LayerNorm(n_rnn) if regularisation else nn.Identity()

        self.glimpse_loss = glimpse_loss
        if self.glimpse_loss:
            self.glimpse_proj_hidden = nn.Linear(n_rnn, int(n_rnn/3))
            self.glimpse_proj_hidden_norm = nn.LayerNorm(int(n_rnn/3)) if regularisation else nn.Identity()
            self.glimpse_proj = nn.Linear(int(n_rnn/3), input_feats)
            self.glimpse_proj_norm = nn.LayerNorm(input_feats) if regularisation else nn.Identity()
            self.glimpse_module = nn.ModuleList([self.glimpse_proj_hidden,self.glimpse_proj_hidden_norm,self.glimpse_proj,self.glimpse_proj_norm])

        self.semantic_loss = semantic_loss
        if self.semantic_loss:
            self.semantic_proj_hidden = nn.Linear(n_rnn, int(n_rnn/3))
            self.semantic_proj_hidden_norm = nn.LayerNorm(int(n_rnn/3)) if regularisation else nn.Identity()
            self.semantic_proj = nn.Linear(int(n_rnn/3), 768)
            self.semantic_proj_norm = nn.LayerNorm(768) if regularisation else nn.Identity()
            self.semantic_module = nn.ModuleList([self.semantic_proj_hidden,self.semantic_proj_hidden_norm,self.semantic_proj,self.semantic_proj_norm])

        self.scene_loss = scene_loss
        if self.scene_loss:
            self.scene_proj_hidden = nn.Linear(n_rnn, int(n_rnn/3))
            self.scene_proj_hidden_norm = nn.LayerNorm(int(n_rnn/3)) if regularisation else nn.Identity()
            self.scene_proj = nn.Linear(int(n_rnn/3), 91)
            self.scene_module = nn.ModuleList([self.scene_proj_hidden,self.scene_proj_hidden_norm,self.scene_proj])

        self.gazeloc_loss = gazeloc_loss
        if self.gazeloc_loss:
            self.gazeloc_proj_hidden = nn.Linear(n_rnn, int(n_rnn/3))
            self.gazeloc_proj_hidden_norm = nn.LayerNorm(int(n_rnn/3)) if regularisation else nn.Identity()
            self.gazeloc_proj = nn.Linear(int(n_rnn/3), 2)
            self.gazeloc_module = nn.ModuleList([self.gazeloc_proj_hidden,self.gazeloc_proj_hidden_norm,self.gazeloc_proj])

    def forward(self, actvs_seq, coord_seq):

        activations = {}

        # actvs_seq = self.backbone(imgs)
        activations['rn50_glimpse'] = actvs_seq

        actv_proj = F.relu(self.actv_proj_norm(self.actv_proj(self.actv_dropout(actvs_seq))))
        if self.input_split == 1:
            B, T, D = actv_proj.shape
            new_actvs_proj = torch.zeros(B, T * 2, D, device=actv_proj.device, dtype=actv_proj.dtype)
            new_actvs_proj[:, ::2, :] = actv_proj

        coord_proj = F.relu(self.coord_proj_norm(self.coord_proj(coord_seq)))
        if self.input_split == 1:
            B, T, D = coord_proj.shape
            new_coord_proj = torch.zeros(B, T * 2, D, device=coord_proj.device, dtype=coord_proj.dtype)
            new_coord_proj[:, 1::2, :] = coord_proj

        activations['joint_proj'] = torch.cat((actv_proj,coord_proj),dim=2) if self.input_split == 0 else torch.cat((new_actvs_proj,new_coord_proj),dim=2)

        if self.return_all_actvs:
            _, seq_len, _ = activations['joint_proj'].size()
            all_hidden_states = [[] for _ in range(self.timestep_multiplier)]
            all_c_states = [[] for _ in range(self.timestep_multiplier)]
            for t in range(seq_len):
                if t == 0:
                    _, (hn, cn) = self.lstm(activations['joint_proj'][:, t].unsqueeze(1))
                else:
                    _, (hn, cn) = self.lstm(activations['joint_proj'][:, t].unsqueeze(1), (hn, cn))
                for layer in range(self.timestep_multiplier):
                    all_hidden_states[layer].append(hn[layer].unsqueeze(1))
                    all_c_states[layer].append(cn[layer].unsqueeze(1))
            for layer in range(self.timestep_multiplier-1):
                activations[f'lstm_h_{layer}'] = torch.cat(all_hidden_states[layer], dim=1) if self.input_split == 0 else torch.cat(all_hidden_states[layer][::2], dim=1) # keep only the image stream
                activations[f'lstm_c_{layer}'] = torch.cat(all_c_states[layer], dim=1) if self.input_split == 0 else torch.cat(all_c_states[layer][::2], dim=1) # keep only the image stream
            lstm_out = torch.cat(all_hidden_states[-1], dim=1)
        else:
            lstm_out, _ = self.lstm(activations['joint_proj'])
        activations['lstm_out'] = self.lstm_norm(lstm_out)
        activations['joint_proj'] = activations['joint_proj'] if self.input_split == 0 else activations['joint_proj'][:,::2] # keep only the image stream

        outputs =  [None for l in range(4)]
        if self.glimpse_loss:
            activations['glimpse_hidden'] = F.relu(self.glimpse_proj_hidden_norm(self.glimpse_proj_hidden(activations['lstm_out'])))
            activations['glimpse_output'] = self.glimpse_proj_norm(self.glimpse_proj(activations['glimpse_hidden']))
            outputs[0] = activations['glimpse_output'] if self.input_split == 0 else activations['glimpse_output'][1::2]
            activations['glimpse_hidden'] = activations['glimpse_hidden'] if self.input_split == 0 else activations['glimpse_hidden'][:,::2]
            activations['glimpse_output'] = activations['glimpse_output'] if self.input_split == 0 else activations['glimpse_output'][:,::2]
        if self.semantic_loss:
            activations['semantic_hidden'] = F.relu(self.semantic_proj_hidden_norm(self.semantic_proj_hidden(activations['lstm_out'])))
            activations['semantic_output'] = self.semantic_proj_norm(self.semantic_proj(activations['semantic_hidden']))
            outputs[1] = activations['semantic_output'] if self.input_split == 0 else activations['semantic_output'][1::2]
            activations['semantic_hidden'] = activations['semantic_hidden'] if self.input_split == 0 else activations['semantic_hidden'][:,::2]
            activations['semantic_output'] = activations['semantic_output'] if self.input_split == 0 else activations['semantic_output'][:,::2]
        if self.scene_loss:
            activations['scene_hidden'] = F.relu(self.scene_proj_hidden_norm(self.scene_proj_hidden(activations['lstm_out'])))
            activations['scene_output'] = self.scene_proj(activations['scene_hidden'])
            outputs[2] = activations['scene_output'] if self.input_split == 0 else activations['scene_output'][1::2]
            activations['scene_hidden'] = activations['scene_hidden'] if self.input_split == 0 else activations['scene_hidden'][:,::2]
            activations['scene_output'] = activations['scene_output'] if self.input_split == 0 else activations['scene_output'][:,::2]
        if self.gazeloc_loss:
            activations['gazeloc_hidden'] = F.relu(self.gazeloc_proj_hidden_norm(self.gazeloc_proj_hidden(activations['lstm_out'])))
            activations['gazeloc_output'] = self.gazeloc_proj(activations['gazeloc_hidden'])
            outputs[3] = activations['gazeloc_output'] if self.input_split == 0 else activations['gazeloc_output'][1::2]
            activations['gazeloc_hidden'] = activations['gazeloc_hidden'] if self.input_split == 0 else activations['gazeloc_hidden'][:,::2]
            activations['gazeloc_output'] = activations['gazeloc_output'] if self.input_split == 0 else activations['gazeloc_output'][:,::2]

        activations['lstm_out'] = activations['lstm_out'] if self.input_split == 0 else activations['lstm_out'][:,1::2] # keep only the saccade stream - that's what prediction is conditioned on.

        if self.return_all_actvs:
            return activations, outputs
        else:
            return outputs
        
class rn18_lstm_gpn(nn.Module):

    def __init__(self, timestep_multiplier=3, glimpse_loss=1, semantic_loss=0, scene_loss=0, gazeloc_loss=0, n_rnn=256, regularisation=1, input_dropout=0.1, rnn_dropout=0.1, return_all_actvs=0, input_split=0, recurrence=True):

        super(rn18_lstm_gpn, self).__init__()

        self.backbone = resnet18()
        self.backbone.maxpool = nn.MaxPool2d(kernel_size=2)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.fc = nn.Identity()

        input_feats = 512
        self.actv_dropout = nn.Dropout(input_dropout) if regularisation else nn.Identity()
        self.actv_proj = nn.Linear(input_feats, int(n_rnn/2))
        self.actv_proj_norm = nn.LayerNorm(int(n_rnn/2)) if regularisation else nn.Identity()
        self.coord_proj = nn.Linear(2, int(n_rnn/2))
        self.coord_proj_norm = nn.LayerNorm(int(n_rnn/2)) if regularisation else nn.Identity()
        self.return_all_actvs = return_all_actvs # this is for analysis
        self.input_split = input_split # 0 if provide image and saccade together and ask for prediction per timestep, 1 if provide image and saccade together and ask for prediction at odd timesteps (when saccade is provided)
        self.recurrence = recurrence # True if recurrence is enabled, False if not

        self.timestep_multiplier = timestep_multiplier
        self.lstm = nn.LSTM(n_rnn,n_rnn,self.timestep_multiplier,dropout = rnn_dropout if regularisation else 0,batch_first=True) if self.recurrence else NonRecurrentLSTM(n_rnn,n_rnn,self.timestep_multiplier,dropout = rnn_dropout if regularisation else 0,batch_first=True)
        self.lstm_norm = nn.LayerNorm(n_rnn) if regularisation else nn.Identity()

        self.glimpse_loss = glimpse_loss
        if self.glimpse_loss:
            self.glimpse_proj_hidden = nn.Linear(n_rnn, int(n_rnn/3))
            self.glimpse_proj_hidden_norm = nn.LayerNorm(int(n_rnn/3))
            self.glimpse_proj = nn.Linear(int(n_rnn/3), input_feats)
            self.glimpse_proj_norm = nn.LayerNorm(input_feats) if regularisation else nn.Identity()
            self.glimpse_module = nn.ModuleList([self.glimpse_proj_hidden,self.glimpse_proj_hidden_norm,self.glimpse_proj,self.glimpse_proj_norm])

        self.semantic_loss = semantic_loss
        if self.semantic_loss:
            self.semantic_proj_hidden = nn.Linear(n_rnn, 256)
            self.semantic_proj_hidden_norm = nn.LayerNorm(256) if regularisation else nn.Identity()
            self.semantic_proj = nn.Linear(256, 768)
            self.semantic_proj_norm = nn.LayerNorm(768) if regularisation else nn.Identity()
            self.semantic_module = nn.ModuleList([self.semantic_proj_hidden,self.semantic_proj_hidden_norm,self.semantic_proj,self.semantic_proj_norm])

        self.scene_loss = scene_loss
        if self.scene_loss:
            self.scene_proj_hidden = nn.Linear(n_rnn, 256)
            self.scene_proj_hidden_norm = nn.LayerNorm(256) if regularisation else nn.Identity()
            self.scene_proj = nn.Linear(256, input_feats)
            self.scene_proj_norm = nn.LayerNorm(input_feats) if regularisation else nn.Identity()
            self.scene_module = nn.ModuleList([self.scene_proj_hidden,self.scene_proj_hidden_norm,self.scene_proj,self.scene_proj_norm])

        self.gazeloc_loss = gazeloc_loss
        if self.gazeloc_loss:
            self.gazeloc_proj_hidden = nn.Linear(n_rnn, int(n_rnn/3))
            self.gazeloc_proj_hidden_norm = nn.LayerNorm(int(n_rnn/3)) if regularisation else nn.Identity()
            self.gazeloc_proj = nn.Linear(int(n_rnn/3), 2)
            self.gazeloc_module = nn.ModuleList([self.gazeloc_proj_hidden,self.gazeloc_proj_hidden_norm,self.gazeloc_proj])

    def forward(self, glimpse_seq=None, coord_seq=None, only_rn=False, input_actvs=False, actvs=None):

        activations = {}

        if not input_actvs:
            actvs_seq = self.backbone(glimpse_seq.reshape(-1,3,91,91))
            actvs_seq = actvs_seq.reshape(glimpse_seq.size(0), glimpse_seq.size(1), -1)
            if only_rn:
                return actvs_seq
            activations['rn50_glimpse'] = actvs_seq[:,:-1,:] # exclude the last one as it has no saccade after it
            actv_proj = F.relu(self.actv_proj_norm(self.actv_proj(self.actv_dropout(actvs_seq[:,:-1,:]))))
        else:
            actvs_seq = actvs
            activations['rn50_glimpse'] = actvs_seq
            actv_proj = F.relu(self.actv_proj_norm(self.actv_proj(self.actv_dropout(actvs_seq))))

        if self.input_split == 1:
            B, T, D = actv_proj.shape
            new_actvs_proj = torch.zeros(B, T * 2, D, device=actv_proj.device, dtype=actv_proj.dtype)
            new_actvs_proj[:, ::2, :] = actv_proj

        coord_proj = F.relu(self.coord_proj_norm(self.coord_proj(coord_seq)))
        if self.input_split == 1:
            B, T, D = coord_proj.shape
            new_coord_proj = torch.zeros(B, T * 2, D, device=coord_proj.device, dtype=coord_proj.dtype)
            new_coord_proj[:, 1::2, :] = coord_proj

        activations['joint_proj'] = torch.cat((actv_proj,coord_proj),dim=2) if self.input_split == 0 else torch.cat((new_actvs_proj,new_coord_proj),dim=2)

        if self.return_all_actvs:
            _, seq_len, _ = activations['joint_proj'].size()
            all_hidden_states = [[] for _ in range(self.timestep_multiplier)]
            all_c_states = [[] for _ in range(self.timestep_multiplier)]
            for t in range(seq_len):
                if t == 0:
                    _, (hn, cn) = self.lstm(activations['joint_proj'][:, t].unsqueeze(1))
                else:
                    _, (hn, cn) = self.lstm(activations['joint_proj'][:, t].unsqueeze(1), (hn, cn))
                for layer in range(self.timestep_multiplier):
                    all_hidden_states[layer].append(hn[layer].unsqueeze(1))
                    all_c_states[layer].append(cn[layer].unsqueeze(1))
            for layer in range(self.timestep_multiplier-1):
                activations[f'lstm_h_{layer}'] = torch.cat(all_hidden_states[layer], dim=1) if self.input_split == 0 else torch.cat(all_hidden_states[layer][::2], dim=1) # keep only the image stream
                activations[f'lstm_c_{layer}'] = torch.cat(all_c_states[layer], dim=1) if self.input_split == 0 else torch.cat(all_c_states[layer][::2], dim=1) # keep only the image stream
            lstm_out = torch.cat(all_hidden_states[-1], dim=1)
        else:
            lstm_out, _ = self.lstm(activations['joint_proj'])
        activations['lstm_out'] = self.lstm_norm(lstm_out)
        activations['joint_proj'] = activations['joint_proj'] if self.input_split == 0 else activations['joint_proj'][:,::2] # keep only the image stream

        outputs =  [None for l in range(4)]
        if self.glimpse_loss:
            activations['glimpse_hidden'] = F.relu(self.glimpse_proj_hidden_norm(self.glimpse_proj_hidden(activations['lstm_out'])))
            activations['glimpse_output'] = self.glimpse_proj_norm(self.glimpse_proj(activations['glimpse_hidden']))
            outputs[0] = activations['glimpse_output'] if self.input_split == 0 else activations['glimpse_output'][1::2]
            activations['glimpse_hidden'] = activations['glimpse_hidden'] if self.input_split == 0 else activations['glimpse_hidden'][:,::2]
            activations['glimpse_output'] = activations['glimpse_output'] if self.input_split == 0 else activations['glimpse_output'][:,::2]
        if self.semantic_loss:
            activations['semantic_hidden'] = F.relu(self.semantic_proj_hidden_norm(self.semantic_proj_hidden(activations['lstm_out'])))
            activations['semantic_output'] = self.semantic_proj_norm(self.semantic_proj(activations['semantic_hidden']))
            outputs[1] = activations['semantic_output'] if self.input_split == 0 else activations['semantic_output'][1::2]
            activations['semantic_hidden'] = activations['semantic_hidden'] if self.input_split == 0 else activations['semantic_hidden'][:,::2]
            activations['semantic_output'] = activations['semantic_output'] if self.input_split == 0 else activations['semantic_output'][:,::2]
        if self.scene_loss:
            activations['scene_hidden'] = F.relu(self.scene_proj_hidden_norm(self.scene_proj_hidden(activations['lstm_out'])))
            activations['scene_output'] = self.scene_proj_norm(self.scene_proj(activations['scene_hidden']))
            outputs[2] = activations['scene_output'] if self.input_split == 0 else activations['scene_output'][1::2]
            activations['scene_hidden'] = activations['scene_hidden'] if self.input_split == 0 else activations['scene_hidden'][:,::2]
            activations['scene_output'] = activations['scene_output'] if self.input_split == 0 else activations['scene_output'][:,::2]
        if self.gazeloc_loss:
            activations['gazeloc_hidden'] = F.relu(self.gazeloc_proj_hidden_norm(self.gazeloc_proj_hidden(activations['lstm_out'])))
            activations['gazeloc_output'] = self.gazeloc_proj(activations['gazeloc_hidden'])
            outputs[3] = activations['gazeloc_output'] if self.input_split == 0 else activations['gazeloc_output'][1::2]
            activations['gazeloc_hidden'] = activations['gazeloc_hidden'] if self.input_split == 0 else activations['gazeloc_hidden'][:,::2]
            activations['gazeloc_output'] = activations['gazeloc_output'] if self.input_split == 0 else activations['gazeloc_output'][:,::2]

        activations['lstm_out'] = activations['lstm_out'] if self.input_split == 0 else activations['lstm_out'][:,1::2] # keep only the saccade stream - that's what prediction is conditioned on.

        if self.return_all_actvs:
            return activations, outputs, actvs_seq
        else:
            return outputs, actvs_seq # need actvs_seq as actvs_seq[:,-1,:] is important for loss computation


###########################################
# New Non-Recurrent LSTM Wrapper Classes  #
###########################################

class NonRecurrentLSTMCell(nn.Module):
    """
    A non-recurrent LSTM cell that computes its activations solely from the current input.
    It mimics the standard LSTM gate computations, but instead of the usual cell update:
      c_t = f_t * c_{t-1} + i_t * g_t,
    it computes:
      c_t = i_t * g_t,
    so that no past state is carried over.

    This cell accepts an optional hidden state tuple (h, c) for drop-in compatibility,
    but ignores it.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(NonRecurrentLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        # This linear layer computes all four gate pre-activations from the current input.
        self.linear = nn.Linear(input_size, 4 * hidden_size, bias=bias)
    
    def forward(self, x, hx=None):
        # x: (batch, input_size); hx is ignored.
        gates = self.linear(x)  # (batch, 4*hidden_size)
        # Split into four gates: input (i), forget (f), candidate (g), output (o)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        # Remove recurrence: ignore c_{t-1} by computing the cell state solely as:
        c = i * g
        h = o * torch.tanh(c)
        return h, c

class NonRecurrentLSTM(nn.Module):
    """
    A wrapper that mimics the nn.LSTM API but removes all recurrence.
    Instead of carrying hidden and cell state across time steps, each time step is processed independently.
    
    Accepts:
      - input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional
      
    It also accepts an optional initial state (hx) and returns a tuple (output, (h_n, c_n))
    to be fully compatible with nn.LSTM.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(NonRecurrentLSTM, self).__init__()
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        # Create layers: each layer may have one cell per direction.
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            # For the first layer, the input size is as provided;
            # for subsequent layers, it is hidden_size * num_directions.
            layer_input_size = input_size if layer == 0 else hidden_size * num_directions
            layer_cells = nn.ModuleList()
            # Forward cell.
            layer_cells.append(NonRecurrentLSTMCell(layer_input_size, hidden_size, bias))
            if bidirectional:
                # Backward cell.
                layer_cells.append(NonRecurrentLSTMCell(layer_input_size, hidden_size, bias))
            self.layers.append(layer_cells)
        
        # Optional dropout applied on the output of each layer (except the last).
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x, hx=None):
        """
        x: Tensor of shape (seq_len, batch, input_size) if batch_first==False,
           or (batch, seq_len, input_size) if batch_first==True.
        hx: Optional tuple (h0, c0). (Ignored in computation.)
        
        Returns:
            output: Tensor of shape (seq_len, batch, hidden_size) for unidirectional,
                    or (seq_len, batch, 2*hidden_size) for bidirectional.
            (h_n, c_n): final hidden and cell states with shape 
                        (num_layers * num_directions, batch, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # Now: (seq_len, batch, input_size)
        seq_len, batch_size, _ = x.size()
        layer_h_n = []
        layer_c_n = []
        output = x

        for layer_idx, layer_cells in enumerate(self.layers):
            if len(layer_cells) == 1:  # Unidirectional.
                cell = layer_cells[0]
                outputs = []
                h_t_last, c_t_last = None, None
                for t in range(seq_len):
                    h_t, c_t = cell(output[t])  # hx is ignored.
                    outputs.append(h_t.unsqueeze(0))
                    h_t_last, c_t_last = h_t, c_t
                layer_output = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_size)
                # The final state for this layer is the last time step.
                h_n = h_t_last.unsqueeze(0)  # (1, batch, hidden_size)
                c_n = c_t_last.unsqueeze(0)  # (1, batch, hidden_size)
            else:  # Bidirectional.
                cell_f, cell_b = layer_cells[0], layer_cells[1]
                outputs_f = []
                outputs_b = []
                h_t_last_f, c_t_last_f = None, None
                h_t_last_b, c_t_last_b = None, None
                for t in range(seq_len):
                    h_f, c_f = cell_f(output[t])
                    outputs_f.append(h_f.unsqueeze(0))
                    h_t_last_f, c_t_last_f = h_f, c_f
                for t in reversed(range(seq_len)):
                    h_b, c_b = cell_b(output[t])
                    outputs_b.insert(0, h_b.unsqueeze(0))
                    # For backward direction, the final state is from the first time step.
                    if t == 0:
                        h_t_last_b, c_t_last_b = h_b, c_b
                out_f = torch.cat(outputs_f, dim=0)    # (seq_len, batch, hidden_size)
                out_b = torch.cat(outputs_b, dim=0)      # (seq_len, batch, hidden_size)
                layer_output = torch.cat([out_f, out_b], dim=2)  # (seq_len, batch, 2*hidden_size)
                h_n_f = h_t_last_f.unsqueeze(0)
                c_n_f = c_t_last_f.unsqueeze(0)
                h_n_b = h_t_last_b.unsqueeze(0)
                c_n_b = c_t_last_b.unsqueeze(0)
                h_n = torch.cat([h_n_f, h_n_b], dim=0)  # (2, batch, hidden_size)
                c_n = torch.cat([c_n_f, c_n_b], dim=0)  # (2, batch, hidden_size)
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            output = layer_output
            layer_h_n.append(h_n)
            layer_c_n.append(c_n)
        # Concatenate final states from all layers along the layer dimension.
        h_n_all = torch.cat(layer_h_n, dim=0)
        c_n_all = torch.cat(layer_c_n, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (h_n_all, c_n_all)
    
# to-do: transformer_gpn