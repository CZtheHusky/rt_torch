import torch
from robotic_transformer import MaxViT, RT1, film_efficientnet_b3
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


# vit = MaxViT(
#     num_classes=1000,
#     dim_conv_stem=64,
#     dim=96,
#     dim_head=32,
#     depth=(2, 2, 5, 2),
#     window_size=7,
#     mbconv_expansion_rate=4,
#     mbconv_shrinkage_rate=0.25,
#     dropout=0.1
# )

effnet = film_efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
# orig_effnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
# film_weight = effnet.state_dict()
# orig_weight = orig_effnet.state_dict()
# test = torch.randn(12, 3, 300, 300)
# effnet.eval()
# orig_effnet.eval()
# res1 = effnet(test)
# res2 = orig_effnet(test)
# assert res1.equal(res2)
model = RT1(efficient=effnet, num_actions=11, depth=6, heads=8, dim_head=64, cond_drop_prob=0.2)
# import pdb; pdb.set_trace()

# video1 = torch.randn(12, 3, 300, 300)
# features = effnet(video1)
video = torch.randn(2, 3, 6, 300, 300).to(device)
inst = ['bring me that apple sitting on the table', 'please pass the butter']

train_logits = model(video, inst)  # (2, 6, 11, 256) # (batch, frames, actions, bins)

# after much training

model.eval()
eval_logits = model(video, inst, cond_scale=3.)  # classifier free guidance with conditional scale of 3
print(eval_logits)
