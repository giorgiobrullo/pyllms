"""
Utility functions for handling images in vision models.
"""

import base64
import os
import requests
from typing import Optional, Union, List


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Encode a local image file to base64.
    
    Returns:
        Tuple of (base64_string, media_type)
    """
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Detect image format from extension
    ext = image_path.lower().split('.')[-1]
    if ext == 'jpg':
        ext = 'jpeg'
    elif ext not in ['jpeg', 'png', 'gif', 'webp']:
        ext = 'jpeg'  # Default fallback
    
    return base64_image, f"image/{ext}"


def download_and_encode_image(url: str) -> tuple[str, str]:
    """
    Download image from URL and encode to base64.
    
    Returns:
        Tuple of (base64_string, media_type)
    """
    # Use pyllms user agent
    headers = {
        'User-Agent': 'pyllms/1.0'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    base64_image = base64.b64encode(response.content).decode('utf-8')
    
    # Try to detect format from URL
    ext = url.lower().split('.')[-1].split('?')[0]
    if ext == 'jpg':
        ext = 'jpeg'
    elif ext not in ['jpeg', 'png', 'gif', 'webp']:
        # Try to detect from content-type header
        content_type = response.headers.get('content-type', '')
        if 'jpeg' in content_type or 'jpg' in content_type:
            ext = 'jpeg'
        elif 'png' in content_type:
            ext = 'png'
        elif 'gif' in content_type:
            ext = 'gif'
        elif 'webp' in content_type:
            ext = 'webp'
        else:
            ext = 'jpeg'  # Default fallback
    
    return base64_image, f"image/{ext}"


def parse_base64_data_url(data_url: str) -> tuple[str, str]:
    """
    Parse a base64 data URL to extract the base64 string and media type.
    
    Returns:
        Tuple of (base64_string, media_type)
    """
    if data_url.startswith('data:'):
        # Format: data:image/jpeg;base64,/9j/4AAQ...
        parts = data_url.split(',', 1)
        if len(parts) == 2:
            header, base64_data = parts
            media_type = header.split(':')[1].split(';')[0]
            return base64_data, media_type
    
    # Assume raw base64 with JPEG as default
    return data_url, "image/jpeg"


def normalize_images_input(images: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """
    Normalize images input to a list.
    
    Args:
        images: Single image or list of images (paths, URLs, or base64)
    
    Returns:
        List of image inputs or None
    """
    if not images:
        return None
    
    if isinstance(images, str):
        return [images]
    
    return images