import torch
import numpy as np
import torch.nn.functional as F



class ActionTokenizer:
    def __init__(
        self, num_action_bin: int = 256, action_path=None, quantile=True,
        ):
        assert action_path is not None
        actions = torch.tensor(np.load(action_path))
        self.mu = actions.mean().item()
        self.sig = actions.std().item()
        actions = (actions - self.mu) / self.sig
        self.max = actions.max()
        self.min = actions.min()
        if quantile:
            q = torch.linspace(0, 1, num_action_bin + 1)
            quantiles = torch.quantile(actions, q, dim=0)
            self.act_dim = actions.shape[1]
            tmp_bucket = quantiles[1:-1]
            self.bucket = [tmp_bucket[:, i].contiguous() for i in range(self.act_dim)]
            self.bins = ((quantiles[:-1] + quantiles[1:]) / 2).contiguous()
        self.num_action_bin = num_action_bin
        self.quantile = quantile

    def discretize(self, x):
        """
        Discretization of float scalars, if is_action then don't need mu-law scaling.
        :param x:
        :param is_action:
        :return:
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.copy()).float()
        x = (x - self.mu) / self.sig
        # import pdb; pdb.set_trace()
        # x = torch.clamp(x, self.min, self.max)
        if not self.quantile:
            x = ((x - self.min) / (self.max - self.min) * (self.num_action_bin - 1)).int().type(torch.int64).long() # x < num_action_bin
        else:
            indices = []
            x = x.transpose(1, 0).contiguous()
            for i in range(self.act_dim):
                indices.append(torch.bucketize(x[i], self.bucket[i]))
            x = torch.stack(indices, dim=1)
        return x

    def discrete2Scalar(self, x):
        if x.shape[-1] != self.num_action_bin:
            action = x
        else:
            action = x.argmax(-1)
        if action.max() >= self.num_action_bin or action.min() < 0:
            print(
                "Warning of exceeded range of discrete number to recontruct, "
                "by default values will be cliped, min: {}, max:{}".format(
                    action.min(), action.max()
                )
            )
            import pdb; pdb.set_trace()
            action = torch.clip(action, 0, self.num_action_bin - 1)
        if self.quantile:
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            action = torch.gather(self.bins, dim=0, index=action)
            # import pdb; pdb.set_trace()
            action = action.squeeze(0).numpy()
        else:
            action = ((action.float()) / self.num_action_bin) * (self.max - self.min) + self.min
        action = action * self.sig + self.mu
        return action

if __name__ == "__main__":
    tokenizer = ActionTokenizer(256, action_path="/raid/robotics_data/language_table_sim_npz/all_actions.npy", quantile=True)
    tokenizer_1 = ActionTokenizer(256, action_path="/raid/robotics_data/language_table_sim_npz/all_actions.npy", quantile=False)
    actions = np.random.uniform(-1, 1, 24) * 0.03
    actions = actions.reshape(12, 2)
    actions_discrete = tokenizer.discretize(torch.tensor(actions))
    recon_actions = tokenizer.discrete2Scalar(actions_discrete)
    delta = recon_actions - actions
    delta1 = tokenizer_1.discrete2Scalar(tokenizer_1.discretize(torch.tensor(actions))) - actions
    import pdb; pdb.set_trace()
    print(delta.mean())
