import torch
import numpy as np
import torch.nn.functional as F



class ActionTokenizer:
    def __init__(
        self, num_action_bin: int = 256, action_max: float = 0.1, action_min: float = -0.1, quantile_path=None    
        ):
        assert action_max == -action_min
        self.num_action_bin = num_action_bin
        if not quantile_path:
            self.max = action_max
            self.min = action_min
            self.quantile = None
        else:
            self.quantile = np.load(quantile_path)
            self.act_dim = self.quantile.shape[1]
            # import pdb; pdb.set_trace()
            tmp_bucket = torch.tensor(self.quantile[1:-1])
            self.bucket = [tmp_bucket[:, i].contiguous() for i in range(self.act_dim)]
            self.bins = torch.tensor((self.quantile[:-1] + self.quantile[1:]) / 2).contiguous()
            self.max = self.quantile[-1]
            self.min = self.quantile[0]
            

    def discretize(self, x):
        """
        Discretization of float scalars, if is_action then don't need mu-law scaling.
        :param x:
        :param is_action:
        :return:
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.copy()).float()
        # import pdb; pdb.set_trace()
        # x = torch.clamp(x, self.min, self.max)
        if self.quantile is None:
            x = ((x - self.min) / self.max / 2 * (self.num_action_bin - 1)).int().type(torch.int64).long() # x < num_action_bin
        else:
            indices = []
            x = x.transpose(1, 0).contiguous()
            for i in range(self.act_dim):
                indices.append(torch.bucketize(x[i], self.bucket[i]))
            x = torch.stack(indices, dim=1)
        return x

    def discrete2Scalar(self, x):
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
        if self.quantile is not None:
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            action = torch.gather(self.bins, dim=0, index=action)
            # import pdb; pdb.set_trace()
            action = action.squeeze(0).numpy()
            
        else:
            action = ((action.float() / self.num_action_bin) * 2 - 1) / 10
        
        return action