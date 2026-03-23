import random
from torchvision import transforms
from PIL import Image, ImageFilter

class RandomGaussianBlur:
    """Apply Gaussian Blur randomly with a given probability."""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img


def random_color_jitter(image):
    color_jitter = transforms.ColorJitter(
        brightness=random.uniform(0, 0.2),
        contrast=random.uniform(0, 0.2),
        saturation=random.uniform(0, 0.2),
        hue=(-0.1, 0.1)  # 使用元组指定范围
    )
    return color_jitter(image)

def get_train_transform_fn(config):

    if config["dataset_augmentations"]['dataset_type'] in ['gfnet_datasets']:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2), 
            transforms.RandomGrayscale(p=0.2),
            RandomGaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(random_color_jitter), #随机调整亮度
            transforms.ToTensor(),              # 将PIL Image或numpy.ndarray转换为tensor，并归一化至[0,1]
            transforms.Normalize((0.5,), (0.5,)), # 使得像素值分布在[-1,1]之间
            ])


    return transform

def get_val_transform_fn(config):

    transform = transforms.Compose([
        transforms.ToTensor(),              # 将PIL Image或numpy.ndarray转换为tensor，并归一化至[0,1]
        transforms.Normalize((0.5,), (0.5,)), # 使得像素值分布在[-1,1]之间
    ])

    return transform

def val_inverse_transform(image_tensor, config):
    transform = get_val_transform_fn(config) 
    normalize_transform = [t for t in transform.transforms if isinstance(t, transforms.Normalize)][0]
    mean = normalize_transform.mean
    std = normalize_transform.std
    
    image = image_tensor.clone()
    
    # 针对每个样本的每个通道进行处理
    for batch_idx in range(image.shape[0]):  # 遍历batch
        for channel_idx in range(image.shape[1]):  # 遍历channel
            image[batch_idx, channel_idx] = image[batch_idx, channel_idx] * std[channel_idx] + mean[channel_idx]
    
    return image