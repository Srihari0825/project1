import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Load and preprocess image
img_path = "icon.png"  # Replace with your file name
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (400, 400))
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

# Parameters
num_pins = 200
num_lines = 3000
radius = 190
center = (200, 200)

# Generate pins on a circle
def generate_pins(n, r, c):
    return [
        (
            int(c[0] + r * math.cos(2 * math.pi * i / n)),
            int(c[1] + r * math.sin(2 * math.pi * i / n))
        )
        for i in range(n)
    ]

# Draw a line and subtract brightness
def draw_line(img, pt1, pt2, value):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.line(mask, pt1, pt2, 255, 1)
    img[mask == 255] = np.clip(img[mask == 255] - value, 0, 255)
    return img, mask

# Optimize next pin selection (local search)
def select_next_pin(working_img, curr_pin, pins, search_radius=20):
    best_score = -np.inf
    best_pin = None
    best_mask = None

    for offset in range(1, search_radius + 1):
        for direction in [-1, 1]:
            target = (curr_pin + direction * offset) % len(pins)
            temp_img = working_img.copy()
            temp_img, mask = draw_line(temp_img, pins[curr_pin], pins[target], 10)
            score = np.sum((working_img - temp_img) * (mask == 255))
            if score > best_score:
                best_score = score
                best_pin = target
                best_mask = mask

    return best_pin, best_mask

# Main execution
pins = generate_pins(num_pins, radius, center)
canvas = np.ones_like(image) * 255
working_img = image.copy()
line_sequence = []
current_pin = 0

for _ in range(num_lines):
    next_pin, mask = select_next_pin(working_img, current_pin, pins)
    if next_pin is not None:
        canvas[mask == 255] = 0
        working_img[mask == 255] = np.clip(working_img[mask == 255] - 10, 0, 255)
        line_sequence.append((current_pin, next_pin))
        current_pin = next_pin

# Show final result
plt.imshow(canvas, cmap='gray')
plt.title("String Art Result")
plt.axis("off")
plt.show()

# Save output
cv2.imwrite("string_art_result.png", canvas)