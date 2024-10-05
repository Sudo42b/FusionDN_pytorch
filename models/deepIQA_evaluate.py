import torch
from torch import nn
from torch.functional import F
import six

class IQA_Model(nn.Module):
    def __init__(self, top:str='patchwise'):
        super(IQA_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)

        self.fc1     = nn.Linear(512, 512)
        self.fc2     = nn.Linear(512, 1)
        
        self.fc1_a   = nn.Linear(512, 512)
        self.fc2_a   = nn.Linear(512, 1)
        self.features = None
        self.top = top
        self.loss = None
        self.y = None
        self.a = None

    def forward(self, x_data:torch.Tensor, 
                y_data:torch.Tensor, 
                n_patches=32):
        # if not isinstance(x_data, Variable):
        #     x = Variable(x_data)
        # else:
        #     x = x_data
        #     x_data = x.data
        self.n_images = y_data.shape[0]
        self.n_patches = x_data.shape[0]
        self.n_patches_per_image = self.n_patches / self.n_images
        x = x_data
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2, 2)
        
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h,2, 2)
        
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pool2d(h,2, 2)
        
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h,2, 2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pool2d(h,2, 2)
        
        h_ = h.detach().clone()
        self.features = h_
        # F.flatten
        
        try:
            ht = torch.flatten(h_, 1)
            h_ = self.fc1(ht)
        except:
            print(h_.shape)
            exit()
        
        h = F.dropout(F.relu(h_), p=0.5)
        h = self.fc2(h)
        
        # features(= h_), h
        if self.top == "weighted":
            a = F.dropout(F.relu(self.fc1_a(h_)), p=0.5)
            a = F.relu(self.fc2_a(a))+0.000001
            t = y_data
            self.weighted_loss(h, a, t)
        elif self.top == "patchwise":
            a = torch.ones_like(h)
            t = y_data.repeat(n_patches)
            # t = torch.repeat(y_data, n_patches)
            self.patchwise_loss(h, a, t)

        if self.training:
            return self.loss
        else:
            return self.loss, self.y, self.a


    def patchwise_loss(self, h, a, t):
        self.loss = torch.sum(torch.abs(h - t.reshape(-1,1)))
        self.loss /= self.n_patches
        if self.n_images > 1:
            h = torch.split(h, self.n_images, 0)
            a = torch.split(a, self.n_images, 0)
        else:
            h, a = [h], [a]
        self.y = h
        self.a = a

    def weighted_loss(self, h, a, t):
        self.loss = 0
        if self.n_images > 1:
            h = torch.split(h, self.n_images, 0)
            a = torch.split(a, self.n_images, 0)
            t = torch.split(t, self.n_images, 0)
        else:
            h, a, t = [h], [a], [t]

        for i in range(self.n_images):
            y = torch.sum(h[i]*a[i], 0) / torch.sum(a[i], 0)
            self.loss += torch.abs(y.to(t[i].device) - torch.reshape(t[i], (1,)))
        self.loss /= self.n_images
        self.y = h
        self.a = a

def extract_patches(arr: torch.Tensor, 
                    patch_shape=(3, 32, 32), 
                    extraction_step=32):
    """
    Extract patches from the input tensor, using PyTorch's (C, H, W) convention.
    
    Args:
    arr (torch.Tensor): Input tensor of shape (C, H, W) or (N, C, H, W)
    patch_shape (tuple): Shape of the patches to extract (C, H, W)
    extraction_step (int or tuple): Stride for patch extraction
    
    Returns:
    torch.Tensor: Extracted patches of shape (out_h, out_w, C, patch_h, patch_w) or (N, out_h, out_w, C, patch_h, patch_w)
    """
    # Handle different input shapes
    if arr.dim() == 3:
        arr = arr.unsqueeze(0)  # Add batch dimension if not present
    N, C, H, W = arr.shape
    # Handle different types of patch_shape and extraction_step
    if isinstance(patch_shape, int):
        patch_shape = (C, patch_shape, patch_shape)
    if isinstance(extraction_step, int):
        extraction_step = (extraction_step, extraction_step)
    
    patch_c, patch_h, patch_w = patch_shape
    stride_h, stride_w = extraction_step
    
    assert C == patch_c, f"Channel dimension mismatch: input has {C} channels, patch expects {patch_c}"
    
    # Use unfold to extract patches
    patches = arr.unfold(2, patch_h, stride_h).unfold(3, patch_w, stride_w)
    
    # Reshape to (N, out_h, out_w, C, patch_h, patch_w)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    
    # Remove batch dimension if input was 3D
    if arr.dim() == 4 and N == 1:
        patches = patches.squeeze(0)
    
    return patches

from utils.util import load_weights

@torch.no_grad()
def IQA(inputs, 
        trained_model_path:str,
        patch_size:int=32,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # input: input image
    # top: choices={'patchwise','weighted'}
    # model: path to the trained model
    model = IQA_Model(top = 'weighted')
    model = load_weights(model, trained_model_path).to(device)
    
    
    model.eval()
    len=inputs.shape[0]
    scores=torch.zeros(len,1)
    for ii in range(len):
        if inputs.dim() == 3:
            inp = inputs[ii].unsqueeze(0)
            input = torch.cat([inp, inp, inp], dim=0)
        else:
            input = torch.cat([inputs[ii], inputs[ii], inputs[ii]], dim=0)
        patches = extract_patches(input,
                                  patch_shape = (3, patch_size, patch_size),
                                  extraction_step = patch_size)
        X = patches.reshape((-1, 3, patch_size, patch_size))
        
        y = []
        weights = []
        batchsize = min(2000, X.shape[0])
        t = torch.zeros((1, 1), dtype=torch.float32)
        model.eval()
        for i in six.moves.range(0, X.shape[0], batchsize):
            X_batch = X[i:i + batchsize]
            # NR:
            loss, pred, a = model(X_batch, t, X_batch.shape[0])
            # FR:
            # X_ref_batch = X_ref[i:i + batchsize]
            # X_ref_batch = xp.array(X_ref_batch.astype(np.float32))
            # model.forward(X_batch, X_ref_batch, t, False, n_patches_per_image = X_batch.shape[0])
            
            y.append(pred[0].reshape((-1,)))
            weights.append(a[0].reshape((-1,)))


            y = torch.concatenate(y)
            weights = torch.concatenate(weights)

        # print("%f" % (np.sum(y * weights) / np.sum(weights)))
        scores[ii] = torch.sum(y * weights) / torch.sum(weights)

    return scores.to(device)



# def EN(inputs:torch.Tensor, patch_size:str=64):
#     len = inputs.shape[0] # batch size
#     entropies = torch.zeros(shape = (len, 1))
#     grey_level = 256
#     counter = torch.zeros(shape = (grey_level, 1))

#     for i in range(len):
#         input_uint8 = (inputs[i, :, :, 0] * 255).astype(torch.uint8)
#         input_uint8 = input_uint8 + 1
#         for m in range(patch_size):
#             for n in range(patch_size):
#                 indexx = input_uint8[m, n]
#                 counter[indexx] = counter[indexx] + 1
#         total = torch.sum(counter)
#         p = counter / total
#         for k in range(grey_level):
#             if p[k] != 0:
#                 entropies[i] = entropies[i] - p[k] * torch.log2(p[k])
#     return entropies

def EN(inputs: torch.Tensor, patch_size: int = 64):
    """
    Calculate entropy for each patch in the input tensor.
    
    Args:
    inputs (torch.Tensor): Input tensor of shape (B, H, W) or (B, C, H, W)
    patch_size (int): Size of the patches to extract
    
    Returns:
    torch.Tensor: Entropy values for each patch, shape (B, 1)
    """
    # Ensure input is 4D (B, C, H, W)
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(1)  # Add channel dimension if not present
    
    B, C, H, W = inputs.shape
    assert C == 1, "Input should have only one channel"
    
    # Extract patches
    patches = F.unfold(inputs, kernel_size=patch_size, stride=patch_size)
    patches = patches.view(B, patch_size*patch_size, -1)  # (B, patch_size^2, num_patches)
    # Convert to uint8 (0-255 range)
    patches_uint8 = (patches * 255).to(torch.uint8)
    
    # Calculate histogram for each patch
    hist = torch.stack([torch.histc(patch.float(), bins=256, min=0, max=255) 
                        for patch in patches_uint8])  # (B, 256, num_patches)
    if hist.dim() == 2:
        hist = hist.unsqueeze(1)
    # Calculate probability distribution
    p = hist / (patch_size * patch_size)
    
    # Avoid log(0) by masking out zero probabilities
    p = p.clamp(min=1e-10)
    
    # Calculate entropy: -sum(p * log2(p))
    entropies = -torch.sum(p * torch.log2(p), dim=1)  # (B, num_patches)
    
    # Average entropy over all patches for each image
    avg_entropies = entropies.mean(dim=1,keepdim=True)  # (B, 1)
    
    return avg_entropies

def W(inputs1:torch.Tensor,
      inputs2:torch.Tensor, 
      trained_model_path:str, 
      w_en:int,
      c:int,
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    iqa1 = IQA(inputs = inputs1.to(device), 
               trained_model_path = trained_model_path)
    iqa2 = IQA(inputs = inputs2.to(device), 
               trained_model_path = trained_model_path)

    en1 = EN(inputs1)
    en2 = EN(inputs2)

    score1 = iqa1 + w_en * en1
    score2 = iqa2 + w_en * en2
    w1 = torch.exp(score1 / c) / (torch.exp(score1 / c) + torch.exp(score2 / c))
    w2 = torch.exp(score2 / c) / (torch.exp(score1 / c) + torch.exp(score2 / c))

    # print('IQA:   1: %f, 2: %f' % (iqa1[0], iqa2[0]))
    # print('EN:    1: %f, 2: %f' % (en1[0], en2[0]))
    # print('total: 1: %f, 2: %f' % (score1[0], score2[0]))
    # print('w1: %s, w2: %s\n' % (w1[0], w2[0]))
    # print('IQA:   1: %f, 2: %f' % (iqa1[1], iqa2[1]))
    # print('EN:    1: %f, 2: %f' % (en1[1], en2[1]))
    # print('total: 1: %f, 2: %f' % (score1[1], score2[1]))
    # print('w1: %s, w2: %s\n' % (w1[1], w2[1]))
    # fig = plt.figure()
    # fig1 = fig.add_subplot(221)
    # fig2 = fig.add_subplot(222)
    # fig3 = fig.add_subplot(223)
    # fig4 = fig.add_subplot(224)
    # fig1.imshow(inputs1[0, :, :, 0], cmap = 'gray')
    # fig2.imshow(inputs2[0, :, :, 0], cmap = 'gray')
    # fig3.imshow(inputs1[1,:,:,0],cmap='gray')
    # fig4.imshow(inputs2[1,:,:,0],cmap='gray')
    # plt.show()
    return {"W1": w1, "W2": w2}
