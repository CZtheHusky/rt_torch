import torch
import numpy as np
import torch.nn.functional as F



class ActionTokenizer:
    def __init__(
        self, num_action_bin: int = 256, action_max: float = 0.1, action_min: float = -0.1,
    ):
        assert action_max == -action_min
        self.num_action_bin = num_action_bin
        self.max = action_max
        self.min = action_min

    def discretize(self, x):
        """
        Discretization of float scalars, if is_action then don't need mu-law scaling.
        :param x:
        :param is_action:
        :return:
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.copy()).float()
        x = torch.clamp(x, -0.1, 0.1)
        # import pdb
        # pdb.set_trace()
        x = ((x - self.min) / self.max / 2 * (self.num_action_bin - 1)).long() # x < num_action_bin
        # import pdb; pdb.set_trace()
        # try:
        #     x = F.one_hot(x, num_classes=self.num_action_bin)
        # except Exception as e:
        #     import pdb
        #     pdb.set_trace()
        return x

    def discrete2Scalar(self, x):
        action = x.argmax(-1)
        if action.max() >= self.num_continuous_bin or action.min() < 0:
            print(
                "Warning of exceeded range of discrete number to recontruct, "
                "by default values will be cliped, min: {}, max:{}".format(
                    action.min(), action.max()
                )
            )
            import pdb
            pdb.set_trace()
            action = torch.clip(action, 0, self.num_continuous_bin - 1)

        action = ((action.float() / self.num_continuous_bin) * 2 - 1) / 10
        return action