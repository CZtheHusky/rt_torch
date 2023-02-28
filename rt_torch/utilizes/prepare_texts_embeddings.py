import os
import numpy as np
from classifier_free_guidance_pytorch import T5Adapter
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"



# parser.add_argument('--check_point', action='store_true')
# parser.add_argument('--reward_shaping', action='store_true')
# parser.add_argument('--mix_maps', action='store_true')

dataset_paths = {
    'language_table': '/raid/robotics_data/language_table_npz',
    'language_table_sim': '/raid/robotics_data/language_table_sim_npz',
    'language_table_blocktoblock_sim': '/raid/robotics_data/language_table_blocktoblock_sim_npz',
    'language_table_blocktoblock_4block_sim': '/raid/robotics_data/language_table_blocktoblock_4block_sim_npz',
    'language_table_blocktoblock_oracle_sim': '/raid/robotics_data/language_table_blocktoblock_oracle_sim_npz',
    'language_table_blocktoblockrelative_oracle_sim': '/raid/robotics_data/language_table_blocktoblockrelative_oracle_sim_npz',
    'language_table_blocktoabsolute_oracle_sim': '/raid/robotics_data/language_table_blocktoabsolute_oracle_sim_npz',
    'language_table_blocktorelative_oracle_sim': '/raid/robotics_data/language_table_blocktorelative_oracle_sim_npz',
    'language_table_separate_oracle_sim': '/raid/robotics_data/language_table_separate_oracle_sim_npz',
}


def main():
    model = T5Adapter(name='google/t5-v1_1-base', device='cuda')
    for k, v in dataset_paths.items():
        if os.path.exists(v):
            obs_path = v + "/observations"
            inst_path = obs_path + "/instructions"
            output_path = os.path.join(obs_path, "inst_embedding_t5")
            os.makedirs(output_path, exist_ok=True)
            files = os.listdir(inst_path)
            for file in tqdm(files):
                file_path = os.path.join(inst_path, file)
                inst = np.load(file_path)['arr_0']
                instruction_embed = model.embed_text(list(inst))
                # import pdb; pdb.set_trace()
                np.savez_compressed(os.path.join(output_path, file), instruction_embed.detach().cpu().numpy())
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
