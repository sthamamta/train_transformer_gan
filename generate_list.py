import os

def get_image_paths(root_dir='classifier_dataset'):
    image_paths = []
    for folder in range(5):  # Assuming subfolders are named 0, 1, 2, 3, and 4
        folder_path = os.path.join('classifier_dataset', str(folder))
        print(folder_path)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_paths.append(os.path.join(folder_path, filename))
    return image_paths

# Replace 'images' with the actual path to your directory containing subfolders 0, 1, 2, 3, 4
image_paths = get_image_paths('images')
print(image_paths)

print(len(image_paths))
# for path in image_paths:
#     print(path)

def save_image_paths(image_paths, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_file = os.path.join(output_folder, 'image_paths.txt')
    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')
