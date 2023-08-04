import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def main(first_folder, second_folder, save_dir):
    # Get the list of image files in the folders
    image_files1 = os.listdir(first_folder)
    image_files2 = os.listdir(second_folder)

    # Sort the image files to ensure they are in the same order
    image_files1.sort()
    image_files2.sort()

    # Loop through the image files
    for i, (file1, file2) in enumerate(zip(image_files1, image_files2)):
        # Read the images from the folders
        image1 = Image.open(os.path.join(first_folder, file1))
        image2 = Image.open(os.path.join(second_folder, file2))
        
        # Calculate the difference image
        difference_image =np.array(image1) - np.array(image2)
        difference_image = Image.fromarray(difference_image)
        
        # Display the images using matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        
        axes[0].imshow(image1, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(first_folder)
        
        axes[1].imshow(image2, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(second_folder)
        
        axes[2].imshow(difference_image, cmap='gray')
        axes[2].axis('off')
        axes[2].set_title('Difference Image')
        
        # Save the figure in the output directory
        output_path = os.path.join(save_dir, f'figure_{i+1}.png')
        plt.savefig(output_path, bbox_inches='tight')
        
        # Close the figure
        plt.close(fig)

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Image Difference')
    
    # Add arguments
    parser.add_argument('--first_folder', type=str,required=False, help='Path to the first image folder', default='images')
    parser.add_argument('--second_folder', type=str,required=False, help='Path to the second image folder', default='images1')
    parser.add_argument('--save_dir', type=str, required=False, help='Path to the save directory', default='difference_images')
    
    # Parse the arguments
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Call the main function with the provided arguments
    main(args.first_folder, args.second_folder, args.save_dir)
