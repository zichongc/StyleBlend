import os
import argparse
import math
from tqdm import tqdm
from typing import Literal
from omegaconf import OmegaConf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.training_utils import cast_training_params

from .pipeline import CustomizedStableDiffusionPipeline
from .utils import TextureStyleTrainingDataset, fst_collate_fn, customize_token_embeddings, save_embeddings


def textual_inversion(
    cfg,
    accelerator,
    pipeline,
    unet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    dataloader,
    placeholder_token_ids,
    token_info,
    style,
    output_dir=None,
):
    num_update_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    cast_training_params(text_encoder, dtype=torch.float32)

    params_to_optim = list(text_encoder.get_input_embeddings().parameters())
    for p in params_to_optim:
        print(p.requires_grad)
    optimizer = AdamW(
        params=params_to_optim,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'constant',
        optimizer,
        num_warmup_steps=50 * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
        num_cycles=1,
        power=1.
    )

    text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, dataloader, lr_scheduler
    )

    # start training
    print("***** Textual Inversion *****")
    print(f"  Total optimization steps = {cfg.max_train_steps}")
    print(f"  Total epochs = {epochs}")
    print(f"  Data samples = {len(dataloader)}")
    print(f"  Style = {style}")

    device = pipeline._execution_device
    global_steps = 0
    progress_bar = tqdm(range(0, cfg.max_train_steps), initial=0, desc='Steps')

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    text_encoder.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            if global_steps >= cfg.max_train_steps:
                break

            with accelerator.accumulate(text_encoder):
                images = batch['images']
                prompts = batch['prompts']
                latents = batch['latents']
                batch_size = len(images)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (batch_size,), device=device
                )
                timesteps = timesteps.long()

                token_ids = tokenizer(prompts, padding='max_length', truncation=True, max_length=77,
                                      return_tensors='pt').input_ids
                encoder_hidden_states = text_encoder(token_ids.to(device))[0].to(latents.dtype)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # make sure no update on any embedding weights except the newly added tokens
                index_no_updates = torch.ones((len(tokenizer),)).bool()
                index_no_updates[min(placeholder_token_ids): max(placeholder_token_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_steps += 1
                logs = {'epoch': epoch + 1, 'step': step + 1, 'loss': f'{loss:.5f}'}
                progress_bar.set_postfix(**logs)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    if output_dir is not None and accelerator.is_main_process:
        # Save the newly trained embeddings
        weight_name = f"{style}_texture_style_embeds.bin"
        save_path = os.path.join(output_dir, weight_name)
        save_embeddings(text_encoder=accelerator.unwrap_model(text_encoder), save_path=save_path, token_info=token_info)


def dreambooth_lora(
    cfg,
    accelerator,
    pipeline,
    unet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    dataloader,
    style,
    output_dir=None,
):
    num_update_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    cast_training_params(unet, dtype=torch.float32)

    params_to_optim = list(filter(lambda p: p.requires_grad, unet.parameters()))
    # print(len(params_to_optim))
    optimizer = AdamW(
        params=params_to_optim,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        name='constant',
        optimizer=optimizer,
        num_warmup_steps=50 * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
        num_cycles=1,
        power=1.
    )

    unet, dataloader, optimizer, lr_scheduler = accelerator.prepare(
        unet, dataloader, optimizer, lr_scheduler
    )

    # start training
    print("***** DreamBooth lora *****")
    print(f"  Total optimization steps = {cfg.max_train_steps}")
    print(f"  Total epochs = {epochs}")
    print(f"  Data samples = {len(dataloader)}")
    print(f'  LoRA rank = {cfg.rank}')
    print(f"  Style = {style}")

    unet.train()
    device = pipeline._execution_device
    global_steps = 0
    progress_bar = tqdm(range(0, cfg.max_train_steps), initial=0, desc='Steps')
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            if global_steps >= cfg.max_train_steps:
                break

            with accelerator.accumulate(unet):
                images, prompts = batch['images'], batch['prompts']
                latents = batch['latents']
                batch_size = len(images)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (batch_size,), device=device
                )
                timesteps = timesteps.long()

                token_ids = tokenizer(prompts, padding='max_length', truncation=True, max_length=77,
                                      return_tensors='pt').input_ids
                encoder_hidden_states = text_encoder(token_ids.to(device))[0]

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())
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

    if output_dir is not None and accelerator.is_main_process:
        # Save the lora weights following the dreambooth-lora code example of JDiffusion
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(unet)))
        weight_name = f"{style}_unet_lora.bin"
        LoraLoaderMixin.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=False,
            weight_name=weight_name
        )
        print('Saved unet lora weights to', os.path.join(output_dir, weight_name))


def train(cfg, style, method: Literal['both', 'ti', 'db'] = 'both'):
    mixed_precision = 'fp16'
    weight_dtype = torch.float16 if mixed_precision == 'fp16' else torch.float32
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.texture_style_training.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    training_cfg = cfg.texture_style_training
    pipeline: CustomizedStableDiffusionPipeline = CustomizedStableDiffusionPipeline.from_pretrained(
        cfg.pretrained_model_path, dtype=torch.float32
    ).to(cfg.device)

    # freeze unnecessary parameters
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    unet, vae = pipeline.unet, pipeline.vae
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    noise_scheduler = DDPMScheduler.from_config(
        os.path.join(cfg.pretrained_model_path, "scheduler/scheduler_config.json"))

    token_info = {'texture_style': (training_cfg.placeholder, training_cfg.init_token)}
    placeholder_token_ids, token_info = customize_token_embeddings(
        tokenizer=tokenizer, text_encoder=text_encoder,
        token_info=token_info,
    )

    images_dir = os.path.join(cfg.sample_dir, style)
    dataset = TextureStyleTrainingDataset(
        data_dir=images_dir,
        image_encoder=vae,
        placeholder=training_cfg.placeholder,
        resolution=training_cfg.resolution,
    )
    dataloader = DataLoader(
        dataset, batch_size=training_cfg.batch_size, shuffle=True, num_workers=0, collate_fn=fst_collate_fn)

    if method in ['ti', 'both']:
        # train embeddings for texture style representation
        ti_cfg = training_cfg.textual_inversion
        textual_inversion(
            ti_cfg,
            accelerator,
            pipeline, unet, tokenizer, text_encoder,
            noise_scheduler, dataloader, placeholder_token_ids, token_info,
            style=style, output_dir=cfg.output_dir
        )
    if method in ['db', 'both']:
        # train lora for texture style representation
        text_encoder.requires_grad_(False)
        lora_cfg = training_cfg.dreambooth_lora
        dreambooth_lora(
            lora_cfg,
            accelerator,
            pipeline, unet, tokenizer, text_encoder,
            noise_scheduler, dataloader,
            style=style, output_dir=cfg.output_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training_config_sd21.yaml')
    parser.add_argument('--style', type=int, default=1)
    args = parser.parse_args()
    args.style = f'{args.style:02d}'

    config = OmegaConf.load(args.config)
    set_seed(config.seed)

    config.output_dir = os.path.join(config.output_dir, args.style)
    os.makedirs(config.output_dir, exist_ok=True)

    config_to_save = OmegaConf.create({
        'seed': config.seed,
        'output_dir': config.output_dir,
        'texture_style_training': config.texture_style_training
    })
    OmegaConf.save(config_to_save, os.path.join(config.output_dir, 'texture_style_training_config.yaml'))
    print(f'Configurations saved to {os.path.join(config.output_dir, "texture_style_training_config.yaml")}')

    train(config, style=args.style)
