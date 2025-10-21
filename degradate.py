import cv2
import numpy as np
import os
from PIL import Image, ImageFilter
from skimage import util
import random


def resolution_reduce(image, scale=0.5):
    height, width = image.shape[:2]
    smaller_image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    restored_image = cv2.resize(smaller_image, (width, height), interpolation=cv2.INTER_CUBIC)
    return restored_image
    

def add_dust_and_scratch(image):
    num_scratches = random.randint(0, 3)
    num_dust = random.randint(10, 20)
    corrupted = image.copy()
    h, w = corrupted.shape[:2]
        
    for _ in range(num_scratches):
        scratch_type = random.choice(['random', 'edge_to_edge'])
        if scratch_type == 'random':
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
        else:
            center_x = random.randint(0, w-1)
            center_y = random.randint(0, h-1)
            
            angle = random.uniform(0, 2 * np.pi)
            
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            intersections = []
            
            if dx != 0:
                t = (w - 1 - center_x) / dx
                y = center_y + t * dy
                if 0 <= y <= h - 1:
                    intersections.append((w - 1, int(y)))

            if dx != 0:
                t = (0 - center_x) / dx
                y = center_y + t * dy
                if 0 <= y <= h - 1:
                    intersections.append((0, int(y)))

            if dy != 0:
                t = (h - 1 - center_y) / dy
                x = center_x + t * dx
                if 0 <= x <= w - 1:
                    intersections.append((int(x), h - 1))
                
            if dy != 0:
                t = (0 - center_y) / dy
                x = center_x + t * dx
                if 0 <= x <= w - 1:
                    intersections.append((int(x), 0))

            if len(intersections) >= 2:
                intersections.sort(key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
                pt1 = intersections[0]
                pt2 = intersections[-1]
            else:
                pt1 = (random.randint(0, w-1), random.randint(0, h-1))
                pt2 = (random.randint(0, w-1), random.randint(0, h-1))
            

        gray_value = random.randint(110, 210)
        color = (gray_value, gray_value, gray_value)
        cv2.line(corrupted, pt1, pt2, color, 1)

    for _ in range(num_dust):
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(1, 2)
        gray_value = random.randint(110, 210)
        color = (gray_value, gray_value, gray_value)
        cv2.circle(corrupted, center, radius, color, -1)
        
    return corrupted


def compress_image(image, quality=20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed = cv2.imdecode(encimg, 1)
    return compressed

    
def add_noise(image, noise_type='gaussian', a=0.05):
    if noise_type == 'gaussian':
        noise = np.random.normal(0, a * 255, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
    elif noise_type == 'salt_pepper':
        noisy_image = util.random_noise(image, mode='s&p', amount=a)
        noisy_image = (255 * noisy_image).astype(np.uint8)
    elif noise_type == 'speckle':
        noise = np.random.randn(*image.shape).astype(np.uint8)
        noisy_image = image + image * noise * a
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Unsupported noise type")
    return noisy_image


def apply_degradation(img):
    # Конвертируем PIL.Image -> NumPy (BGR)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if random.random() < 0.4:
        combined_restoration = add_dust_and_scratch(img)
    else:
        combined_restoration = img.copy()

    n = random.random()
    if n < 0.5:
        combined_restoration = add_noise(combined_restoration, 'speckle', np.random.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))
    elif n < 0.75:
        combined_restoration = add_noise(combined_restoration, 'salt_pepper', np.random.choice([0.005, 0.01, 0.015, 0.02]))
    else:
        combined_restoration = add_noise(combined_restoration, 'gaussian', np.random.choice([0.0015, 0.002, 0.001]))

    n = random.random()
    if n < 0.33:
        combined_restoration = resolution_reduce(combined_restoration, np.random.choice([0.3, 0.4, 0.5]))
    elif n < 0.66:
        combined_restoration = compress_image(combined_restoration, np.random.choice([20, 30]))

    # Возвращаем обратно NumPy -> PIL.Image (RGB)
    combined_restoration = cv2.cvtColor(combined_restoration, cv2.COLOR_BGR2RGB)
    combined_restoration = Image.fromarray(combined_restoration)

    return combined_restoration
