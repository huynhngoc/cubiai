import json


# change setttings here

filename = 'normal_abnormal20_scratch'
# dataset filename
ds_files = 'normal_abnormal20.h5'

# how did you resize the images
input_size = 800
# lower learning rates for pretrain models
learning_rates = [0.005, 0.01]
# which EfficientNet
model_types = ['B3', 'B4']
# how many classes, pretrain or from scratch
num_class = 2
pretrain = False


base_ds_path = '/mnt/project/ngoc/CubiAI/datasets/'
if pretrain:
    if num_class <= 2:
        template_fn = 'config/local/pretrain.json'
    else:
        template_fn = 'config/local/pretrain_multiclass.json'
else:
    if num_class <= 2:
        template_fn = 'config/local/scratch.json'
    else:
        template_fn = 'config/local/scratch_multiclass.json'

train_type = 'pretrain' if pretrain else 'scratch'

with open(template_fn, 'r') as f:
    template = json.load(f)

# update batch size and cache
template['dataset_params']['config']['filename'] = base_ds_path + ds_files
template['dataset_params']['config']['batch_size'] = 16
template['dataset_params']['config']['batch_cache'] = 8

for lr in learning_rates:
    lr_str = f'{lr:6.5f}'.split('.')[-1]
    for eff_type in model_types:
        # update input size
        template['input_params']['shape'][0] = input_size
        template['input_params']['shape'][1] = input_size

        # update preprocessors
        template['dataset_params']['config']['preprocessors'][0]['config']['num_class'] = num_class

        # update model architecture
        template['architecture']['num_class'] = num_class
        template['architecture']['pretrained'] = pretrain
        template['architecture']['class_name'] = eff_type

        # update learning rate
        template['model_params']['optimizer']['config']['learning_rate'] = lr

        model = eff_type.lower()

        with open(f'config/{train_type}/{model}_{filename}_lr_{lr_str}.json', 'w') as f:
            json.dump(template, f)
