import h5py
import numpy as np
import torch

path = './pretrained/nr_tid_weighted.model'
# './pretrained/nr_tid.model'의 값들을 pytorch로 이식
from models.deepIQA_evaluate import IQA_Model
model = IQA_Model(top = 'weighted')
res = dict(h5py.File(path, 'r'))
for k, v in res.items():
    # Parsing this one v variable
    for k2, v2 in v.items():
        
        name = k
        print(k2, v2.shape)
        if k2 == 'W':
            name += '.weight'
            # print(v2[()])
            
            model.state_dict()[name].copy_(torch.tensor(v2[()]))
            assert model.state_dict()[name].shape == v2.shape
            assert np.allclose(model.state_dict()[name].numpy(), v2[()])
        elif k2 == 'b':
            # print(v2[()])
            name += '.bias'
            model.state_dict()[name].copy_(torch.tensor(v2[()]))
            assert model.state_dict()[name].shape == v2.shape
            assert np.allclose(model.state_dict()[name].numpy(), v2[()])
        else:
            raise ValueError('Unknown key')
        print('=====================')
    
torch.save(model.state_dict(), './pretrained/nr_tid_weighted.pth')

# model = Model(top = 'weighted')
# serializers.load_hdf5(trained_model_path, model)