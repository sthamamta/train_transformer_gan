
# outputs paths(checkpoints, epoch images,input_batch_images and output batch images) for corresponding datset_name_size_factor_model_name
def set_outputs_dir(opt):
    if opt.patch:
        opt.save_checkpoints_dir = 'outputs/{}/{}/checkpoints/patch/patch-{}/factor_{}/'.format(opt.model_name,opt.exp_name, opt.patch_size,opt.factor)
        opt.epoch_images_dir ='outputs/{}/{}/epoch_images/patch/patch-{}/factor_{}/'.format(opt.model_name,opt.exp_name, opt.patch_size,opt.factor)
        opt.input_images_dir ='outputs/{}/{}/input_images/patch/patch-{}/factor_{}/'.format(opt.model_name, opt.exp_name, opt.patch_size,opt.factor)
        opt.output_images_dir ='outputs/{}/{}/output_images/patch/patch-{}/factor_{}/'.format(opt.model_name, opt.exp_name, opt.patch_size,opt.factor) 
        opt.train_images_dir ='outputs/{}/{}/train_images/patch/patch-{}/factor_{}/'.format(opt.model_name, opt.exp_name, opt.patch_size,opt.factor) 
    else:
        opt.save_checkpoints_dir = 'outputs/{}/{}/checkpoints/factor_{}/'.format(opt.model_name,opt.exp_name,opt.factor)
        opt.epoch_images_dir ='outputs/{}/{}/epoch_images/factor_{}/'.format(opt.model_name,opt.exp_name,opt.factor)
        opt.input_images_dir ='outputs/{}/{}/input_images/factor_{}/'.format(opt.model_name, opt.exp_name,opt.factor)
        opt.output_images_dir ='outputs/{}/{}/output_images/factor_{}/'.format(opt.model_name,opt.exp_name,opt.factor)
        opt.train_images_dir ='outputs/{}/{}/train_images/factor_{}/'.format(opt.model_name,opt.exp_name,opt.factor)

# training metric paths
def set_training_metric_dir(opt):
    if opt.patch:
        opt.loss_dir = 'outputs/{}/{}/losses/patch/patch-{}/factor_{}/'.format(opt.model_name, opt.exp_name, opt.patch_size,opt.factor)
        opt.grad_norm_dir ='outputs/{}/{}/grad_norm/patch/patch-{}/factor_{}/'.format(opt.model_name,opt.exp_name, opt.patch_size,opt.factor)
    else:
        opt.loss_dir = 'outputs/{}/{}/losses/factor_{}/'.format(opt.model_name,opt.exp_name,opt.factor)
        opt.grad_norm_dir ='outputs/{}/{}/grad_norm/factor_{}/'.format(opt.model_name,opt.exp_name,opt.factor)


#plots path
def set_plots_dir(opt):
    if opt.patch:
        opt.plot_dir = 'outputs/{}/{}/plots/patch/patch-{}/factor_{}/'.format(opt.model_name, opt.exp_name,opt.patch_size,opt.factor)
    else:
        opt.plot_dir = 'outputs/{}/{}/plots/factor_{}/'.format(opt.model_name, opt.exp_name,opt.factor)

