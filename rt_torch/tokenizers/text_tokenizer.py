from classifier_free_guidance_pytorch.t5 import T5Adapter
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub



class UniversalSentenceEncoder():
    def __init__(self, 
                 name=None,
                 device=None,
                 ) -> None:
        # module_url = "/home/cz/universal-sentence-encoder_4"
        # self.model = hub.load(module_url)
        self.model = SentenceTransformer('/home/cz/distiluse-base-multilingual-cased-v1').to(device)
        self.device =  device

    @property
    def dim_latent(self):
        return 512
    
    @torch.no_grad()
    def embed_text(
        self,
        texts: List[str],
    ):
        # import pdb; pdb.set_trace()
        encoded_text = self.model.encode(texts)
        # import pdb
        # pdb.set_trace()
        return torch.tensor(encoded_text).to(self.device)
    
class UniversalSentenceEncoderTF():
    def __init__(self, 
                 name=None,
                 device=None,
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
        if name == 't5':
            self.text_model = T5Adapter(None, device)
            self.text_embed_dim = 768
        elif name == "use":
            self.text_model = UniversalSentenceEncoder(name, device)
            self.text_embed_dim = 512
        elif name == "use_tf":
            self.text_model = UniversalSentenceEncoderTF(name, device)
            self.text_embed_dim = 512
        else:
            raise NotImplementedError
        # self.text_model = None

        self.device = device


    def embed_texts(self, texts: List[str], device):
        # device = self.device
        # print(f"embedding {device}")
        text_embeds = []
        text_embed = self.text_model.embed_text(texts)
        text_embeds.append(text_embed.to(device))
        return torch.cat(text_embeds, dim = -1)