import os, torch
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from .flux_degrade import degradation_bsrgan_plus
import wandb
class myutil:
    def __init__(self):
        super().__init__()
        pass

    def imread_uint(self,image_path):
        image = Image.open(image_path)
        image_np = np.array(image)
        return image_np
    def imsave(self,  image,image_path):
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image)
    def uint2single(self,img):
        return np.float32(img / 255.)
    def single2uint(self,img):
        return np.uint8((img.clip(0, 1) * 255.).round())
from datetime import datetime
class PreImage:
    def __init__(self, hflip=True, vflip=True,rot90=True):
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90


    def __call__(self, image,p=0.3,p90=0.2,resize=True):
        do_hflip =self.hflip and random.random() < p
        do_vflip =self.vflip and random.random() < p
        do_rot90 =self.rot90 and random.random() < p90
        def _augment(img):
            if do_hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                # PIL 原生垂直翻转（等价于 numpy 的 img[::-1, :, :]）
            if do_vflip:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                # PIL 原生90度顺时针旋转（等价于 numpy 的 img.transpose(1,0,2)）
            if do_rot90:
                img = img.transpose(Image.ROTATE_90)
            return img

        def get_random_crop_params(pil_img, output_size):
            """
            等价于 transforms.RandomCrop.get_params()
            :param pil_img: PIL.Image对象
            :param output_size: 裁剪尺寸 (h, w)
            :return: i, j, h, w （左上角坐标i,j，裁剪高h，裁剪宽w）
            """
            w, h = pil_img.size  # PIL的size是 (宽, 高) → 对应torch的 W, H
            th, tw = output_size
            # 核心随机逻辑：和torchvision源码完全一致，有效范围内随机取左上角坐标
            if w == tw and h == th:
                return 0, 0, h, w
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            return i, j, th, tw
        if resize:

            random_num = random.sample(range(6), 1)[0]

            if random_num % 6 == 1:
                target_size = (384, 384)
            elif random_num % 6 == 2:
                target_size = (512, 512)
            elif random_num % 6 == 3:
                target_size = (768, 768)
            elif random_num % 6 == 4:
                target_size = (1024,1280)
            elif random_num % 6 == 5:
                target_size=(768,1920)
            else:
                pass
            if random_num:
                i, j, h, w = get_random_crop_params(image, output_size=target_size)
                # 2. PIL的crop方法裁剪：核心API Image.crop( (左, 上, 右, 下) )
                image = image.crop((j, i, j + w, i + h))
                # clear = clear.crop((j, i, j + w, i + h))
                # image =image.resize(
                # size=target_size,  # (W, H)，和 cv2 顺序一致
                # resample=Image.Resampling.BILINEAR  # 对应 cv2.INTER_LINEAR
                # # 可选：更高画质用 Image.Resampling.LANCZOS（对应 cv2.INTER_LANCZOS4）
                # )
        image = _augment(image)
        # time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]s
        # save_path=os.path.join('/data/guyf/codes/diffsynth/examples/qwen_image/model_training/lora/temp',time_str+'.jpg')
        image=image.convert('RGB')
        return image
def preimage_collate_fn(
        batch,
        image_keys=("image", "edit_image"),  # 需要处理的图片字段
        preimage_kwargs={"hflip": True, "vflip": True, "rot90": True},  # PreImage 初始化参数
        sync_augment=True,
        add_noise=True,# 是否同步成对图片的增强逻辑（关键！避免moire/clear修改不一致）
):
    preprocessor = PreImage(**preimage_kwargs)
    modified_batch = batch.copy()[0]
    # 遍历批次中的每个样本
    myuti=myutil()
    seed = random.getstate() if sync_augment else None
    for sample in batch:
        # 【关键】如果同步增强，先固定随机种子（保证同一样本的所有图片修改一致）
        # 遍历样本中的每个字段
        for key in sample.keys():
            # 处理图片字段：执行 PreImage 增强+缩放
            if key in image_keys:
                img = sample[key]
                # print('pre',type(img_aug))Image
                # print("img_aug像素最大值：", np.array(img_aug).max())255
                if sync_augment:
                    random.setstate(seed)
                img_aug =preprocessor(img,resize=True)
                # img_aug =preprocessor(img,resize=False)
                if add_noise and key=="edit_image" and random.random() < 0.3:
                    img_aug=np.array(img_aug)
                    img_aug=myuti.uint2single(img_aug)
                    img_aug,_=degradation_bsrgan_plus(img_aug,sf=1)
                    img_aug=myuti.single2uint(img_aug)
                    img_aug=Image.fromarray(img_aug)
                # time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # save_path=os.path.join('/data/guyf/codes/diffsynth_2/examples/qwen_image/model_training/lora/temp',key+time_str+'.jpg')
                # img_aug.save(save_path)


                modified_batch[key]=img_aug
            # 非图片字段：直接保留原始值
            else:
                modified_batch[key]=sample[key]
        # assert False
    final_batch=modified_batch
    # assert False
    return final_batch
def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda batch: preimage_collate_fn(
        batch,
        image_keys=("image", "edit_image"),  # 处理 moire/clear 字段
        preimage_kwargs={"hflip": True, "vflip": True, "rot90": True},
        sync_augment=True,
        add_noise=True),
                                             num_workers=num_workers)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        # logger.info("Logging some dataset samples.")
        formatted_images = []
        formatted_control_images = []
        all_prompts = []
        for i, batch in enumerate(dataloader):
            images = batch["image"]
            control_images = batch["edit_image"]
            prompts = batch["prompt"]

            if len(formatted_images) > 4:
                break

            # for img, control_img, prompt in zip(images, control_images, prompts):
            formatted_images.append(images)
            formatted_control_images.append(control_images)
            all_prompts.append(prompts)

        logged_artifacts = []
        for img, control_img, prompt in zip(formatted_images, formatted_control_images, all_prompts):
            logged_artifacts.append(wandb.Image(control_img, caption="Conditioning"))
            logged_artifacts.append(wandb.Image(img, caption=prompt))

        wandb_tracker = [tracker for tracker in accelerator.trackers if tracker.name == "wandb"]
        wandb_tracker[0].log({"dataset_samples": logged_artifacts})
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator=accelerator, model=model,lr=scheduler.get_last_lr()[0],
                                         loss=loss.detach().item(), save_steps=save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
