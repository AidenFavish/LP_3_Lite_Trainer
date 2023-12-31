import os
import csv
import random
from PIL import Image, ImageDraw


def generate_images(data, folder_path, csv_filename):
    """
    Function to generate images and a CSV file with the image details.
    Each image will have a random background color and a randomly positioned and sized square with a random color.
    """

    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, csv_filename), mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['frame number', 'centerX', 'centerY', 'radius', 'turn'])

        for i in data:
            # Random background color
            bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new('RGB', (1000, 1000), color=bg_color)

            # Random square properties
            square_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            top_left_x = i[0]
            top_left_y = i[1]
            square_size = i[2]
            center_x = top_left_x + square_size / 2
            center_y = top_left_y + square_size / 2

            # Draw the square
            draw = ImageDraw.Draw(img)
            draw.rectangle([top_left_x, top_left_y, top_left_x + square_size, top_left_y + square_size],
                           fill=square_color)

            # Save the image
            img_name = f'frame{i}.0.jpg'
            img.save(os.path.join(folder_path, img_name))

            # Write the square's details to the CSV
            writer.writerow([i, center_x / 1000.0, center_y / 1000.0, square_size / 2000.0, 0])
            # Averages: centerX = 0.5001, centerY = 0.5001, radius = 0.0625

    print(f"Generated {len(data)} images in folder '{folder_path}' with CSV file '{csv_filename}'")


def complete_random(types, copies):
    points = []
    types_of_points = []

    for i in range(types):
        square_size = random.randint(50, 200)
        top_left_x = random.randint(0, 1000 - square_size)
        top_left_y = random.randint(0, 1000 - square_size)
        types_of_points.append([top_left_x, top_left_y, square_size])

    for i in range(copies):
        for p in types_of_points:
            points.append(p)

    return points


def discrete_location(types, copies):
    points = []

    for i in range(copies):
        for j in range(len(types)):
            square_size = random.randint(50, 200)
            top_left_x = types[j][0] - square_size / 2
            top_left_y = types[j][1] - square_size / 2
            points.append([top_left_x, top_left_y, square_size])

    return points


generate_images(discrete_location([[250, 500], [750, 500]], 400), "/Users/aiden/Desktop/Training/Training1", "data.csv")
generate_images(discrete_location([[250, 500], [750, 500]], 3), "/Users/aiden/Desktop/Training/Validation1", "data.csv")
