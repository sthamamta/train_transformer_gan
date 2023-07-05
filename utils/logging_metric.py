import pickle
import os


class LogMetric(object):
    def __init__(self,training_dict=None):
        if training_dict:
            self.training_dict = training_dict
        else:
            self.training_dict = { 'train_loss' : []}
        self.val_dict={'val_loss': [], 'val_l1': [],'val_psnr':[],'val_ssim':[],'val_hfen':[],'epoch':[]}
        self.log_dict ={}

    def update_dict_with_key(self,key,value,training=True):
        if training:
            self.training_dict[key].append(value)
        else:
            self.val_dict[key].append(value)

    def update_dict(self,value=[],training=True):
        for (k, v) in zip(self.get_dict_keys(training), value):
            if training:
                self.training_dict[k].append(v)
            else:
                self.val_dict[k].append(v)

    def get_dict_keys(self,training=True):
        if training:
            return list(self.training_dict.keys()) 
        else:
            return list(self.val_dict.keys()) 


    def update_log_dict(self):
        self.log_dict = {**self.training_dict, **self.val_dict}
        return self.log_dict

    def save_dict(self,opt,save_name='loss_metric.pkl'):
        if not os.path.exists(opt.loss_dir):
            os.makedirs(opt.loss_dir)
        save_path=os.path.join(opt.loss_dir,save_name)
        
        self.update_log_dict()
        with open(save_path,"wb") as fp:
            pickle.dump(self.log_dict,fp)
        return save_path  
    


def create_loss_meters_gan():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def create_loss_meters_srdense():
    train_loss = AverageMeter()
    return {'train_loss': train_loss}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), n=count)


def update_epoch_losses(epoch_losses, count,values=[]):
    for (loss_meter,val) in zip(epoch_losses.keys(),values):
        epoch_losses[loss_meter].update(val,n=count)


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count