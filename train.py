import argparse
import os

import torch
import torchvision.transforms.functional as tvt_fn
import tqdm
from torch.cuda.amp import autocast

import wandb
import x_clip
from x_clip.tokenizer import SimpleTokenizer

from x_clip.dataset import ImageTextDataset

def clip_model_from_args():
    return x_clip.CLIP(
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 49408,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        visual_enc_depth = 6,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_heads = 8,
        use_all_token_embeds = True,            # whether to use fine-grained contrastive learning (FILIP)
        decoupled_contrastive_learning = True,  # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
        extra_latent_projection = True,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_visual_ssl = True,                  # whether to do self supervised learning on iages
        visual_ssl_type = 'simclr',             # can be either 'simclr' or 'simsiam', depending on using DeCLIP or SLIP
        use_mlm = False,                        # use masked language learning (MLM) on text (DeCLIP)
        text_ssl_loss_weight = 0.05,            # weight for text MLM loss
        image_ssl_loss_weight = 0.05 
    )
    
def check_args(args):
    if not os.path.exists(r"G:\Datasets\MyFloorplans\text2image\01_data\v1\paired"):
        raise ValueError('Dataset path does not exist')
    if os.path.exists('./checkpoints/x_clip.pt') and not args.overwrite:
        raise ValueError(
            'Checkpoint path already exists, use --overwrite to overwrite')
    else:
        os.makedirs(os.path.dirname('./checkpoints/x_clip.pt'), exist_ok=True)
        
def main():
    #wandb_run = wandb.init(project='x-clip', config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _tokenizer = SimpleTokenizer()

    clip_model = clip_model_from_args()
    clip_model.to(device)
    clip_model.train()

    # clamp to ln(100)
    torch.clamp(clip_model.temperature, min=0.0, max=4.6052)

    def preprocess_fn(x):
        return tvt_fn.resize(tvt_fn.to_tensor(x),
                             (256, 256))

    image_text_dataset = ImageTextDataset(preprocess=preprocess_fn,
                                          folder=r"G:\Datasets\MyFloorplans\text2image\01_data\v1\paired",
                                          bpe_tokenizer=_tokenizer)
    image_text_dataloader = torch.utils.data.DataLoader(
        image_text_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0)

    # Create optimizer
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-4)

    # Train
    for epoch in tqdm.tqdm(range(100)):
        current_epoch_pbar = tqdm.tqdm(enumerate(image_text_dataloader), 
                                       total=len(image_text_dataloader), unit='batch', unit_scale=8)
        for batch_idx, (text, images) in current_epoch_pbar:
            with autocast(enabled=True):
                text, images = map(lambda t: t.cuda(), (text, images))
                mask = torch.ones_like(text).bool()
                loss = clip_model(text, images, text_mask=mask, return_loss=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                current_epoch_pbar.set_description(f'batch {batch_idx} loss {loss:.4f}')
                wandb_run.log({'loss': loss, 'epoch': epoch}, step=batch_idx)

        torch.save(clip_model.state_dict(), './checkpoints/x_clip.pt')
        
if __name__ == "__main__":
    main()