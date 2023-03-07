from classifier_free_guidance_pytorch.t5 import T5Adapter
from typing import List, Optional
import torch
import tensorflow_hub as hub

class UniversalSentenceEncoder():
    def __init__(self, 
                 name,
                 device,
                 ) -> None:
        module_url = "/home/cz/universal-sentence-encoder_4"
        self.model = hub.load(module_url)
        self.device =  device

    @property
    def dim_latent(self):
        return 512
    
    def embed_text(
        self,
        texts: List[str],
    ):
        encoded_text = self.model(texts)
        # import pdb
        # pdb.set_trace()
        return torch.tensor(encoded_text.numpy()).to(self.device)
    


class TextTokenizer():
    def __init__(self,
                 name=None,
                 device=None,
                 ) -> None:
        # if name == 't5':
        #     self.text_model = T5Adapter(None, device)
        # else:
        #     self.text_model = UniversalSentenceEncoder(name, device)
        self.text_model = None
        self.device = device


    def embed_texts(self, texts: List[str]):
        device = self.device
        text_embeds = []
        text_embed = self.text_model.embed_text(texts)
        text_embeds.append(text_embed.to(device))
        return torch.cat(text_embeds, dim = -1)