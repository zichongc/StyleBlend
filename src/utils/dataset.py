import os
import random
from PIL import Image
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


class TextureStyleTrainingDataset(Dataset):
    def __init__(self, data_dir, image_encoder, placeholder: str = '<T>', resolution=768, ):
        super().__init__()
        self.data_dir = data_dir
        self.resolution = resolution
        self.placeholder = placeholder
        self.image_encoder = image_encoder
        # self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)

        self.class_prompts, self.images_pil = self.load_all_data()
        self.n_samples = len(self.images_pil)
        self.latents = [self.encode_image(img) for img in self.images_pil]
        
    def load_all_data(self):
        files = os.listdir(self.data_dir)

        images = []
        class_prompts = []
        for file in files:
            image = Image.open(os.path.join(self.data_dir, file)).convert('RGB').resize(
                (self.resolution, self.resolution))
            class_prompt = file.split('.')[0].replace('_', ' ').lower()
            class_prompts.append(class_prompt)
            images.append(image)

        return class_prompts, images

    def __len__(self):
        return self.n_samples

    @staticmethod
    def aan(string: str):
        return 'an' if string[0].lower() in 'aeiou' else 'a'

    @torch.no_grad()
    def encode_image(self, image: Image.Image):
        if isinstance(image, Image.Image):
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=.5, std=.5)
            ])(image).unsqueeze(0).to(device=self.image_encoder.device, dtype=self.image_encoder.dtype)

        latent = self.image_encoder.encode(image).latent_dist.sample()
        latent = latent * self.image_encoder.config.scaling_factor
        return latent

    def __getitem__(self, index):
        image = self.images_pil[index]
        cls_prompt = self.class_prompts[index]
        latent = self.latents[index]

        prompt = f'{self.aan(cls_prompt).title()} {cls_prompt} in the style of {self.placeholder}.'
        return {'image': image, 'prompt': prompt, 'latent': latent}


def fst_collate_fn(examples):
    prompts = [example['prompt'] for example in examples]
    images = [example['image'] for example in examples]
    latents = torch.cat([example['latent'] for example in examples])
    batch = {'images': images, 'prompts': prompts, 'latents': latents}
    return batch


class CompositionStyleTrainingDataset(Dataset):
    def __init__(self, style_data_dir, plausible_data_dir, image_encoder, placeholder: str = '<C>', resolution=768, ):
        super().__init__()
        self.style_data_dir = style_data_dir
        self.plausible_data_dir = plausible_data_dir
        self.image_encoder = image_encoder
        self.resolution = resolution
        self.class_prompts, self.images = self.load_style_data()

        self.placeholder = placeholder
        self.class_prompt_index = {cp: i for i, cp in enumerate(self.class_prompts)}

        self.plausible_images = self.load_generated_data()
        self.latents = [self.encode_image(img) for img in self.images]

        self.plausible_latents = [[] for _ in range(len(self.class_prompts))]
        for i in range(len(self.plausible_images)):
            latents = [self.encode_image(img) for img in self.plausible_images[i]]
            self.plausible_latents[i] = latents

    def load_style_data(self):
        files = os.listdir(self.style_data_dir)

        images = []
        class_prompts = []
        for file in files:
            image = Image.open(os.path.join(self.style_data_dir, file)).convert('RGB').resize(
                (self.resolution, self.resolution))
            class_prompt = file.split('.')[0].split('-')[0].replace('_', ' ').lower()
            class_prompts.append(class_prompt)
            images.append(image)

        return class_prompts, images

    def load_generated_data(self):
        files = os.listdir(self.plausible_data_dir)
        images = [[] for _ in range(len(self.class_prompts))]

        for file in files:
            image = Image.open(os.path.join(self.plausible_data_dir, file)).convert('RGB').resize(
                (self.resolution, self.resolution))
            class_prompt = file.split('.')[0].split('-')[0].replace('_', ' ').lower()
            images[self.class_prompt_index[class_prompt]].append(image)

        return images

    @staticmethod
    def aan(string: str):
        return 'an' if string[0].lower() in 'aeiou' else 'a'

    @torch.no_grad()
    def encode_image(self, image: Image.Image):
        if isinstance(image, Image.Image):
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=.5, std=.5),
            ])(image).unsqueeze(0).to(device=self.image_encoder.device, dtype=self.image_encoder.dtype)

        latent = self.image_encoder.encode(image).latent_dist.sample()
        latent = latent * self.image_encoder.config.scaling_factor
        return latent

    def __len__(self):
        return len(self.class_prompts)

    def __getitem__(self, index):
        prompt = self.aan(self.class_prompts[index]).title() + ' ' + self.class_prompts[index] + f' in the style of {self.placeholder}.'
        latent = self.latents[index]

        plausible_prompt = self.aan(self.class_prompts[index]).title() + ' ' + self.class_prompts[index] + '.'
        plausible_latent = random.choice(self.plausible_latents[self.class_prompt_index[self.class_prompts[index]]])
        return {'prompt': prompt, 'latent': latent,
                'src_prompt': plausible_prompt, 'src_latent': plausible_latent}


def cst_collate_fn(examples):
    prompts = [example['prompt'] for example in examples]
    latents = torch.cat([example['latent'] for example in examples])
    src_prompts = [example['src_prompt'] for example in examples]
    src_latents = torch.cat([example['src_latent'] for example in examples])
    batch = {'latents': latents, 'prompts': prompts,
             'src_latents': src_latents, 'src_prompts': src_prompts}
    return batch


class ImageFilter:
    """
    Filter images based on their structural (cosine) similarity to the reference image. The CLIP image encoder is
    used to assess the structural similarity between the reference style image and the generated images.
     **Note that we use the embeddings of patch tokens for feature representation, rather than the class embeddings.**
    """

    def __init__(
            self,
            pretrained_model_path,
    ):
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_path)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model_path)

    def __call__(self, target: Image.Image, candidates: List[Image.Image], select_n: int):
        assert select_n <= len(candidates)
        if select_n == len(candidates):
            return candidates

        # get image features and prompt features
        inputs_target = self.processor(text='', images=target, return_tensors="pt")
        inputs_candidates = self.processor(text='', images=candidates, return_tensors="pt")

        with torch.no_grad():
            outputs_target = self.clip_model(output_hidden_states=True, **inputs_target)
            outputs_candidates = self.clip_model(output_hidden_states=True, **inputs_candidates)

        outputs_target = outputs_target.vision_model_output.hidden_states[-1]
        b, _, _ = outputs_target[:, 1:].shape
        target_features = outputs_target[:, 1:].reshape(b, -1)

        outputs_candidates = outputs_candidates.vision_model_output.hidden_states[-1]
        b, _, _ = outputs_candidates[:, 1:].shape
        candidates_features = outputs_candidates[:, 1:].reshape(b, -1)

        sims = F.cosine_similarity(target_features, candidates_features, dim=1)
        # print('similarity scores: ', sims)

        # select the instances with top-n scores
        _, topk_indices = torch.topk(sims, select_n)
        # print(topk_indices)

        selected_instances = [candidates[i] for i in topk_indices]

        return selected_instances
