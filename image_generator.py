import os
import csv
import random
from PIL import Image, ImageDraw


def generate_images(x, folder_path, csv_filename):
    """
    Function to generate images and a CSV file with the image details.
    Each image will have a random background color and a randomly positioned and sized square with a random color.
    """
    scatter_mode = True
    if x is int:
        num_images = x
        scatter_mode = False
    else:
        num_images = len(x)

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
            if scatter_mode:
                top_left_x = x[i][0] - square_size // 2
                top_left_y = x[i][1] - square_size // 2
                center_x = x[i][0]
                center_y = x[i][1]
            else:
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
            writer.writerow([i, center_x / 1000.0, center_y / 1000.0, square_size // 2 / 1000.0, 0])
            # Averages: centerX = 0.5001, centerY = 0.5001, radius = 0.0625

    print(f"Generated {num_images} images in folder '{folder_path}' with CSV file '{csv_filename}'")


def scatter_v1():
    points = []
    size = 250
    for v in range((size // 2)):
        currV = 8 * v
        ctr = 0
        for t in range(currV):
            if ctr + 1 == (size // 2 + 1) // (v + 1):
                if (0 <= t <= v) or (7 * v <= t < 8 * v):
                    x = size // 2 + v
                    y = t + size // 2 if t <= v else size // 2 - (8 * v - t)
                elif v < t < 3 * v:
                    x = t - 2 * v + size // 2
                    y = size // 2 + v
                elif 3 * v <= t <= 5 * v:
                    x = size // 2 - v
                    y = 4 * v - t + size // 2
                else:
                    x = 6 * v - t + size // 2
                    y = size // 2 - v
                points.append((10 * x, 10 * y))
                ctr = -1
            ctr += 1

    # Filter
    points = [[point[0] - 725, point[1] - 725] for point in points if 725 <= point[0] <= 1775 and 725 <= point[1] <= 1775]

    return points


generate_images(x=scatter_v1(), folder_path=r'/home/penny/Desktop/LP_Training/TrainingData2', csv_filename='data.csv')
generate_images(x=10, folder_path=r'/home/penny/Desktop/LP_Training/ValidationData2', csv_filename='data.csv')
