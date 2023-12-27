import os
import csv
import random
from PIL import Image, ImageDraw

"""
Function to generate images and a CSV file with the image details.
Each image will have a random background color and a randomly positioned and sized square with a random color.
"""
def generate_images(num_images, folder_path, csv_filename):
    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, csv_filename), mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['frame number', 'centerX', 'centerY', 'radius', 'turn'])

        for i in range(num_images):
            # Random background color
            bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new('RGB', (1000, 1000), color=bg_color)

            # Random square properties
            square_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            square_size = random.randint(50, 200)
            top_left_x = random.randint(0, 1000 - square_size)
            top_left_y = random.randint(0, 1000 - square_size)
            center_x = top_left_x + square_size // 2
            center_y = top_left_y + square_size // 2

            # Draw the square
            draw = ImageDraw.Draw(img)
            draw.rectangle([top_left_x, top_left_y, top_left_x + square_size, top_left_y + square_size],
                           fill=square_color)

            # Save the image
            img_name = f'frame{i}.0.jpg'
            img.save(os.path.join(folder_path, img_name))

            # Write the square's details to the CSV
            writer.writerow([i, center_x / 1000, center_y / 1000, square_size // 2 / 1000, 0])

    print(f"Generated {num_images} images in folder '{folder_path}' with CSV file '{csv_filename}'")


# Example usage
generate_images(num_images=5, folder_path='/Users/aiden/Desktop/Validation1', csv_filename='data.csv')
