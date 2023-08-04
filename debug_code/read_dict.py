import pickle


# read and append the folder path to dictionary

def read_dictionary(path):
    with open(path,'rb') as f:
        annotation_dict = pickle.load(f)
    return annotation_dict

def save_dicionary(path,annotation_dict):
    with open(path,'wb') as f:
        pickle.dump(annotation_dict, f)
        return 1

path = 'outputs/srdense/srdense_factor_2_mse_loss_bicubic/losses/patch/patch-200/factor_2/loss_metric.pkl'
d_i = read_dictionary(path)
print(d_i)