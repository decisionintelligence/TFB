import torch
import torch.nn as nn


class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5):
        super(TopKGating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

    def decompostion_tp(self, x, alpha=10):
        # x [batch_size, seq_len]
        output = torch.zeros_like(x)
        # [batch_size]
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        # [batch_size, num_expert]
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask] = alpha * torch.log(x[mask] + 1)
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)
        # Ablation Spare MoE
        # output[mask] = 0
        # [batch_size, seq_len]
        return output

    def forward(self, x):
        # [batch_size, seq_len]

        x = self.gate(x)
        clean_logits = x
        # [batch_size, num_experts]

        if self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.decompostion_tp(logits)
        gates = self.softmax(logits)

        # random order
        # indices = torch.randperm(gates.size(0))
        # shuffled_gates = gates[indices]

        # average
        # value = 1.0 / x.shape[1]
        # gates = torch.full(x.shape, value, device=x.device)

        return gates


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class AMS(nn.Module):
    def __init__(self, input_shape, pred_len, ff_dim=2048, dropout=0.2, loss_coef=1.0, num_experts=4, top_k=2):
        super(AMS, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.num_experts = num_experts
        self.top_k = top_k
        self.pred_len = pred_len

        self.gating = TopKGating(input_shape[0], num_experts, top_k)

        self.experts = nn.ModuleList(
            [Expert(input_shape[0], pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)])
        self.loss_coef = loss_coef
        assert (self.top_k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, time_embedding):
        # [batch_size, feature_num, seq_len]
        batch_size = x.shape[0]
        feature_num = x.shape[1]
        # [feature_num, batch_size, seq_len]
        x = torch.transpose(x, 0, 1)
        time_embedding = torch.transpose(time_embedding, 0, 1)

        output = torch.zeros(feature_num, batch_size, self.pred_len).to(x.device)
        loss = 0

        for i in range(feature_num):
            input = x[i]
            time_info = time_embedding[i]
            # x[i]  [batch_size, seq_len]
            gates = self.gating(time_info)

            # expert_outputs [batch_size, num_experts, pred_len]
            expert_outputs = torch.zeros(self.num_experts, batch_size, self.pred_len).to(x.device)

            for j in range(self.num_experts):
                expert_outputs[j, :, :] = self.experts[j](input)
            expert_outputs = torch.transpose(expert_outputs, 0, 1)
            # gates [batch_size, num_experts, pred_len]
            gates = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)
            # batch_output [batch_size, pred_len]
            batch_output = (gates * expert_outputs).sum(1)
            output[i, :, :] = batch_output

            importance = gates.sum(0)
            loss += self.loss_coef * self.cv_squared(importance)

        # [feature_num, batch_size, seq_len]
        output = torch.transpose(output, 0, 1)
        # [batch_size, feature_num, seq_len]

        return output, loss