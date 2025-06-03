import sys
import math
import base64
import numpy as np
import cv2

def clip(val, low, high):
    if val < low:
        return low 
    elif val > high:
        return high 
    else:
        return val


def transform_image_to_base64(image):
    # Convert the image to a base64 string
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text




# Resize and binarize mask array for interpretable segmentation mask
def resize_mask(maskparams, target_width, target_height):
    src = maskparams.get_mask_array() # Retrieve mask array
    if src.size == 0:
        # print("Mask array is None, returning empty array")
        return np.empty((target_height, target_width), dtype=np.uint8)

    dst = np.empty((target_height, target_width), src.dtype) # Initialize array to store re-sized mask
    original_width = maskparams.width
    original_height = maskparams.height
    ratio_h = float(original_height) / float(target_height)
    ratio_w = float(original_width) / float(target_width)
    threshold = maskparams.threshold
    channel = 1

    # Resize from original width/height to target width/height 
    for y in range(target_height):
        for x in range(target_width):
            x0 = float(x) * ratio_w
            y0 = float(y) * ratio_h
            left = int(clip(math.floor(x0), 0.0, float(original_width - 1.0)))
            top = int(clip(math.floor(y0), 0.0, float(original_height - 1.0)))
            right = int(clip(math.ceil(x0), 0.0, float(original_width - 1.0)))
            bottom = int(clip(math.ceil(y0), 0.0, float(original_height - 1.0)))

            for c in range(channel):
                # H, W, C ordering
                # Note: lerp is shorthand for linear interpolation
                left_top_val = float(src[top * (original_width * channel) + left * (channel) + c])
                right_top_val = float(src[top * (original_width * channel) + right * (channel) + c])
                left_bottom_val = float(src[bottom * (original_width * channel) + left * (channel) + c])
                right_bottom_val = float(src[bottom * (original_width * channel) + right * (channel) + c])
                top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left)
                bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left)
                lerp = top_lerp + (bottom_lerp - top_lerp) * (y0 - top)
                if (lerp < threshold): # Binarize according to threshold
                    dst[y,x] = 0
                else:
                    dst[y,x] = 255
    return dst
