import pickle
import os
import numpy as np

# load the dictionary for an real dataset
filename = '../model_bias_experiment/mri_dataset_50/lr_hr_dictionary.pkl'

with open(filename, 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)  # This will print the loaded dictionary

# quit();
# loaded_data = {v: k for k, v in loaded_data.items()}  ### inverting a dictionary

# output_dict = {}
# for lr_key, hr_value in loaded_data.items():
#     # Extract the relevant parts and construct the new key-value pairs
#     lr_parts = lr_key.split('_')
#     hr_parts = hr_value.split('_')
    
#     # Extract the relevant parts and construct the new key-value pairs
#     lr_f_value = lr_parts[1] + '_' + lr_parts[-1].split('.')[0]
#     hr_f_value = hr_parts[1] + '_' + hr_parts[-1].split('.')[0]
    
#     output_dict[lr_f_value] = hr_f_value

# # Print the new dictionary
# # print(output_dict)

# # output_filename = 'output_dict.pkl'
# output_filename = '../model_bias_experiment/mri_dataset_50/lr_hr_index_dictionary.pkl'

# # Save the dictionary to a pickle file
# with open(output_filename, 'wb') as pickle_file:
#     pickle.dump(output_dict, pickle_file)


# with open(output_filename, 'rb') as file:
#     loaded_data = pickle.load(file)

# print(loaded_data)  # This will print the loaded dictionary


# load 25 micron array and truncate start and end 
# check the array range
# normalize and plot

def load_data_nii(array_path):
    import nibabel as nib
    img = nib.load(array_path)
    affine_mat=img.affine
    hdr=img.header
    data = img.get_fdata()
    data_norm = data
    return data_norm

# f1_path = '../array_data/25/f1_25.nii'
# array_path = '../array_data/25/f1_25.nii'
# array_path = '../array_data/25/'

# labels_list = os.listdir(array_path)



# for index,file_array in enumerate(labels_list):

#     if file_array == 'f4_25.nii':
#         pass
#     else:
#         file_path =  os.path.join(array_path, file_array)
#         single_array = load_data_nii(file_path)
#         single_array = single_array[:,:,20:261]
#         if index == 0:
#             array_full = single_array
#         else:
#             array_full =  np.concatenate((array_full, single_array), axis=2)
        
#         print(array_full.shape)
#         print(index, file_array)

labels_array_dir = '../array_data/25/'
inputs_array_dir = '../array_data/50/'

labels_list = os.listdir(labels_array_dir)
input_list = os.listdir(inputs_array_dir)

hr_array_full = [[[]]]
lr_array_full = [[[]]]
index_tuple = {}
count = 0
for index,(file_label_array,file_input_array)  in enumerate(zip(labels_list, input_list)):
    if file_label_array == 'f4_25.nii':
        pass
    else:
        label_file_path =  os.path.join(labels_array_dir, file_label_array)
        single_label_array = load_data_nii(label_file_path)
        single_input_array =  load_data_nii(os.path.join(inputs_array_dir, file_input_array) )
        if index == 0:
            hr_array_full = single_label_array
            lr_array_full = single_input_array
        else:
            hr_array_full =  np.concatenate((hr_array_full, single_label_array), axis=2)
            lr_array_full =  np.concatenate((lr_array_full, single_input_array), axis=2)
        

        index_tuple [file_label_array.split('_')[0]] = count
        count +=1
        print(hr_array_full.shape)
        print(index, file_label_array)


print(index_tuple)

hr_image_index = []
lr_image_index = []
for key, value in loaded_data.items():
    lr_index = int(key.split('_')[4].split('.')[0])
    hr_index = int(value.split('_')[4].split('.')[0])

    multiplier = int(index_tuple[key.split('_')[1]] )

    print(key.split('_')[1], multiplier)

    lr_index = lr_index + (multiplier*152)
    hr_index = hr_index + (multiplier*304)
    hr_image_index.append(hr_index)
    lr_image_index.append(lr_index)

print(hr_image_index)
print(lr_image_index)

print(hr_array_full.shape)
print(lr_array_full.shape)
# print(array_full[:,:,25].shape)

# array_f1 = load_data_nii(f1_path)

# print(array_f1.shape)
# print(array_f1[:,:,20:261].shape)


# # Find the minimum and maximum values along each dimension
# min_along_axis_0 = np.min(array_full, axis=0)
# max_along_axis_0 = np.max(array_full, axis=0)

# min_along_axis_1 = np.min(array_full, axis=1)
# max_along_axis_1 = np.max(array_full, axis=1)

# min_along_axis_2 = np.min(array_full, axis=2)
# max_along_axis_2 = np.max(array_full, axis=2)

# print("Minimum along axis 0:")
# print(min_along_axis_0)
# print("\nMaximum along axis 0:")
# print(max_along_axis_0)

# print("\nMinimum along axis 1:")
# print(min_along_axis_1)
# print("\nMaximum along axis 1:")
# print(max_along_axis_1)

# print("\nMinimum along axis 2:")
# print(min_along_axis_2)
# print("\nMaximum along axis 2:")
# print(max_along_axis_2)