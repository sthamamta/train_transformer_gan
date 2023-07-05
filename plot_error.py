import os
import matplotlib.pyplot as plt
from PIL import Image

def plot_image_patch(folder_directory, patch_size, output_directory):
    num_images = len(folder_directory)
    num_columns = int(num_images ** 0.5) + 1
    num_rows = (num_images // num_columns) + 1

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(10, 10))
    files = ['lr_f1_160_z_71.png']

    for i, folder in enumerate(folder_directory):
        # Get image files in the folder
        # files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if len(files) == 0:
            continue

        # Read the first image
        image_path = os.path.join(folder, files[0])
        img = Image.open(image_path)

        # Extract a patch from the image
        print(img.size)
        x_indx= 150
        y_idx = 450
        patch = img.crop((x_indx, y_idx, x_indx+patch_size, y_idx+patch_size))

        # Plot the patch in the subplot
        ax = axs[i // num_columns, i % num_columns]
        ax.imshow(patch, cmap='gray')
        ax.axis('off')

        # Add directory name as title
        directory_name = os.path.basename(folder)
        ax.set_title(directory_name)

    # Hide any empty subplots
    if num_images < num_rows * num_columns:
        for j in range(num_images, num_rows * num_columns):
            axs[j // num_columns, j % num_columns].axis('off')

    plt.tight_layout()

    # Save the subplot as an image
    output_path = os.path.join(output_directory, "subplot.png")
    plt.savefig(output_path)
    plt.close()

# Example usage
# folder_directory = ["results/bicubic_upsample", "results/bilinear_upsample", "results/nearest_upsample", "results/l1_loss", "results/mse_loss", "results/ssim_loss"]
folder_directory = ["results/bicubic_upsample", "results/bilinear_upsample", "results/nearest_upsample", 
"results/l1_loss", "results/mse_loss", "results/ssim_loss", "test_plot_image/25_micron", "results/ls_25_50_micron"]


patch_size = 200
output_directory = "results/patch_plots"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

plot_image_patch(folder_directory, patch_size, output_directory)
