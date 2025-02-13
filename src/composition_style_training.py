import os
import argparse
import math
import random
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
import diffusers
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.training_utils import cast_training_params
from diffusers.loaders import LoraLoaderMixin
import transformers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from .pipeline import CustomizedStableDiffusionPipeline, SDEditPipeline
from .utils import customize_token_embeddings, save_embeddings, cst_collate_fn, CompositionStyleTrainingDataset, ImageFilter


class AssistantDataset(Dataset):
    def __init__(self, style: str, data_dir: str, resolution=768):
        self.style = style
        self.resolution = resolution
        self.image_dir = os.path.join(data_dir, style)
        self.class_prompts, self.images, self.filenames = self.load_data()

    def load_data(self):
        files = os.listdir(self.image_dir)

        images = []
        class_prompts = []
        filenames = []
        for file in files:
            image = Image.open(os.path.join(self.image_dir, file)).convert('RGB').resize(
                (self.resolution, self.resolution))
            class_prompt = file.split('.')[0].replace('_', ' ').lower()
            class_prompts.append(class_prompt)
            images.append(image)
            filenames.append(file)

        return class_prompts, images, filenames

    @staticmethod
    def aan(string: str):
        return 'an' if string[0].lower() in 'aeiou' else 'a'

    def __len__(self):
        return len(self.class_prompts)

    def __getitem__(self, index):
        prompt = self.aan(self.class_prompts[index]).title() + ' ' + self.class_prompts[index] + ', realistic.'
        return prompt, self.images[index], self.filenames[index]


class PlausibleDataCreator:
    """
    Generate a set of plausible images using SDEdit to create a dataset for composition style learning.
    """
    def __init__(
        self,
        pretrained_sd_model_path,
        pretrained_clip_model=None,
        k=10,
        select_n=10,
        resolution=768,
        device='cuda',
        **kwargs,
    ):
        self.sd_pipe = SDEditPipeline.from_pretrained(pretrained_sd_model_path, dtype=torch.float16).to(device)
        self.sd_pipe.scheduler = DDIMScheduler.from_config(
            os.path.join(pretrained_sd_model_path, 'scheduler/scheduler_config.json'))
        if pretrained_clip_model is not None:
            self.image_filter = ImageFilter(pretrained_clip_model)

        self.k = k
        self.select_n = select_n
        self.resolution = resolution

    def generate(self, style, source_dir, output_dir):
        os.makedirs(os.path.join(output_dir, style), exist_ok=True)
        dataset = AssistantDataset(style, source_dir, resolution=self.resolution)

        for prompt, image, filename in dataset:
            print(filename, prompt)
            candidates = []
            strengths = [random.choice([0.85, 0.8, 0.75]) for _ in range(self.k)]
            print('strengths', strengths)

            for s in strengths:
                candidate = self.sd_pipe(
                    images=[image],
                    prompt=prompt,
                    strength=s,
                    num_inference_steps=20,
                    negative_prompt='ugly, low quality, blur, distorted'
                ).images[0]
                candidates.append(candidate)

            if self.image_filter is not None:
                candidates = self.image_filter(image, candidates, select_n=self.select_n)

            for i, img in enumerate(candidates):
                img.save(os.path.join(output_dir, style, filename[:-4] + f'-{i:03d}.png'))

        print(f'Generated images saved to {output_dir}.')


def create_plausible_dataset(cfg, style):
    sample_dir = cfg.sample_dir
    cst_cfg = cfg.composition_style_training
    output_dir = cst_cfg.dir_to_save_dataset

    creator = PlausibleDataCreator(
        pretrained_sd_model_path=cfg.pretrained_model_path, device=cfg.device,
        **cst_cfg
    )
    creator.generate(style=style, source_dir=sample_dir, output_dir=output_dir)
    del creator


def train(
    cfg,
    style,
):
    mixed_precision = 'fp16'
    weight_dtype = torch.float16 if mixed_precision == 'fp16' else torch.float32
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.composition_style_training.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    training_cfg = cfg.composition_style_training
    pipe: CustomizedStableDiffusionPipeline = CustomizedStableDiffusionPipeline.from_pretrained(
        cfg.pretrained_model_path, torch_dtype=torch.float32
    ).to(cfg.device)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    token_info = {'composition_style': (training_cfg.placeholder, training_cfg.init_token)}
    placeholder_token_ids, token_info = customize_token_embeddings(
        tokenizer=tokenizer, text_encoder=text_encoder,
        token_info=token_info,
        mean_init=False
    )

    unet, vae = pipe.unet, pipe.vae
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    noise_scheduler = DDPMScheduler.from_config(
        os.path.join(cfg.pretrained_model_path, 'scheduler/scheduler_config.json'))

    resolution = training_cfg.resolution
    style_data_dir = os.path.join(cfg.sample_dir, style)
    plausible_data_dir = os.path.join(training_cfg.dir_to_save_dataset, style)
    
    dataset = CompositionStyleTrainingDataset(
        style_data_dir=style_data_dir, plausible_data_dir=plausible_data_dir,
        image_encoder=vae, resolution=resolution, placeholder=training_cfg.placeholder
    )

    dataloader = DataLoader(dataset, training_cfg.batch_size, shuffle=True, collate_fn=cst_collate_fn)

    text_lora_config = LoraConfig(
        r=training_cfg.rank,
        lora_alpha=training_cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    text_encoder.add_adapter(text_lora_config)
    lora_params_to_optimize = list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

    optimizer = AdamW(
        params=lora_params_to_optimize,
        lr=training_cfg.lr,
        betas=(training_cfg.adam_beta1, training_cfg.adam_beta2),
        weight_decay=training_cfg.adam_weight_decay,
        eps=training_cfg.adam_epsilon
    )

    lr_scheduler = get_scheduler(
        'constant',
        optimizer,
        num_warmup_steps=50 * accelerator.num_processes,
        num_training_steps=training_cfg.max_train_steps * accelerator.num_processes,
        num_cycles=1,
        power=1.
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / training_cfg.gradient_accumulation_steps)
    epochs = math.ceil(training_cfg.max_train_steps / num_update_steps_per_epoch)

    cast_training_params(text_encoder, dtype=torch.float32)

    text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, dataloader, lr_scheduler
    )

    # start training
    print("***** Composition Style Training *****")
    print(f"  Total optimization steps = {training_cfg.max_train_steps}")
    print(f"  Total epochs = {epochs}")
    print(f"  Data samples = {len(dataloader)}")
    print(f'  LoRA rank = {training_cfg.rank}')
    print(f"  Style = {style}")

    global_steps = 0
    device = pipe._execution_device
    progress_bar = tqdm(range(0, training_cfg.max_train_steps), initial=0, desc='Steps')

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            if global_steps >= training_cfg.max_train_steps:
                break

            with accelerator.accumulate(text_encoder):
                latents, prompts = batch['latents'], batch['prompts']
                src_latents, src_prompts = batch['src_latents'], batch['src_prompts']
                batch_size = len(latents)

                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (batch_size,), device=device
                )
                timesteps = timesteps.long()
                src_noise = torch.randn_like(src_latents)

                token_ids = tokenizer(prompts, padding='max_length', truncation=True, max_length=77,
                                      return_tensors='pt').input_ids
                encoder_hidden_states = text_encoder(token_ids.to(device))[0].to(latents.dtype)

                src_noisy_latents = noise_scheduler.add_noise(src_latents, src_noise, timesteps)
                src_model_pred = unet(src_noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    src_target = src_noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    src_target = noise_scheduler.get_velocity(src_latents, src_noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                loss = torch.nn.functional.mse_loss(src_model_pred.float(), src_target.float())
                # loss.backward()
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_steps += 1
                logs = {'epoch': epoch + 1, 'step': step + 1, 'loss': f'{loss:.5f}'}
                progress_bar.set_postfix(**logs)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Save the newly trained embeddings
        weight_name = f"{style}_composition_style_embeds.bin"
        save_path = os.path.join(cfg.output_dir, weight_name)
        save_embeddings(text_encoder=accelerator.unwrap_model(text_encoder), save_path=save_path, token_info=token_info)

        # save the lora weights of text encoder
        text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(accelerator.unwrap_model(text_encoder), save_embedding_layers=False))

        weight_name = f"{style}_text_encoder_lora.bin"
        LoraLoaderMixin.save_lora_weights(
            save_directory=cfg.output_dir,
            text_encoder_lora_layers=text_encoder_lora_state_dict,
            safe_serialization=False,
            weight_name=weight_name
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training_config_sd21.yaml')
    parser.add_argument('--style', type=int, default=11)
    parser.add_argument('--create_data', type=str, default='f')
    args = parser.parse_args()
    args.style = f'{args.style:02d}'
    args.create_data = False if 'f' in args.create_data.lower() else True

    config = OmegaConf.load(args.config)

    set_seed(config.seed)

    config.output_dir = os.path.join(config.output_dir, args.style)
    os.makedirs(config.output_dir, exist_ok=True)

    config_to_save = OmegaConf.create({
        'seed': config.seed,
        'output_dir': config.output_dir,
        'composition_style_training': config.composition_style_training
    })
    OmegaConf.save(config_to_save, os.path.join(config.output_dir, 'composition_style_training_config.yaml'))
    print(f'Configurations saved to {os.path.join(config.output_dir, "composition_style_training_config.yaml")}')

    # create training data
    if args.create_data:
        create_plausible_dataset(config, style=args.style)
    # composition style training
    train(config, style=args.style)
