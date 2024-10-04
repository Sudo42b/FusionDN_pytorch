from torch import nn
import torch
from losses import Fro_LOSS
from losses import Per_LOSS
import numpy as np
from numpy.lib.stride_tricks import as_strided
import numbers

def extract_patches(arr, patch_shape = (32, 32, 3), extraction_step = 32):
	arr_ndim = arr.ndim

	if isinstance(patch_shape, numbers.Number):
		patch_shape = tuple([patch_shape] * arr_ndim)
	if isinstance(extraction_step, numbers.Number):
		extraction_step = tuple([extraction_step] * arr_ndim)

	patch_strides = arr.strides

	slices = tuple(slice(None, None, st) for st in extraction_step)
	indexing_strides = arr[slices].strides

	patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
	                       np.array(extraction_step)) + 1

	shape = tuple(list(patch_indices_shape) + list(patch_shape))
	strides = tuple(list(indexing_strides) + list(patch_strides))

	patches = as_strided(arr, shape = shape, strides = strides)
	return patches

def IQA(inputs, trained_model_path):
	# input: input image
	# top: choices={'patchwise','weighted'}
	# model: path to the trained model
	model = Model(top = 'weighted')
	xp = cuda.cupy
	serializers.load_hdf5(trained_model_path, model)
	model.to_gpu()

	len=inputs.shape[0]
	scores=np.zeros(shape=(len,1))
	for ii in range(len):
		input = np.concatenate((inputs[ii,:,:,:], inputs[ii,:,:,:], inputs[ii,:,:,:]), axis = -1)
		patches = extract_patches(input)
		X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))

		# ref_img = np.concatenate((ref_img, ref_img, ref_img), axis = -1)
		# patches = extract_patches(ref_img)
		# X_ref = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))
		y = []
		weights = []
		batchsize = min(2000, X.shape[0])
		t = xp.zeros((1, 1), np.float32)
		for i in six.moves.range(0, X.shape[0], batchsize):
			X_batch = X[i:i + batchsize]
			X_batch = xp.array(X_batch.astype(np.float32))

			# NR:
			model.forward(X_batch, t, False, X_batch.shape[0])
			# FR:
			# X_ref_batch = X_ref[i:i + batchsize]
			# X_ref_batch = xp.array(X_ref_batch.astype(np.float32))
			# model.forward(X_batch, X_ref_batch, t, False, n_patches_per_image = X_batch.shape[0])

			y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
			weights.append(xp.asnumpy(model.a[0].data).reshape((-1,)))


			y = np.concatenate(y)
			weights = np.concatenate(weights)

		# print("%f" % (np.sum(y * weights) / np.sum(weights)))
		scores[ii] = np.sum(y * weights) / np.sum(weights)

	return scores

def EN(inputs, patch_size=16):
    batch_size = inputs.shape[0]
    grey_level = 256

    # Convert inputs to uint8 and add 1
    inputs_uint8 = (inputs[:, 0, :, :] * 255).to(torch.uint8) + 1

    # Flatten the inputs and compute the histogram
    inputs_flat = inputs_uint8.view(batch_size, -1)
    counter = torch.stack([torch.bincount(inputs_flat[i], minlength=grey_level).float() for i in range(batch_size)])

    # Compute probabilities
    total = counter.sum(dim=1, keepdim=True)
    p = counter / total

    # Compute entropies
    entropies = -(p * torch.log2(p + 1e-10)).sum(dim=1, keepdim=True)  # Adding a small value to avoid log(0)

    return entropies

def W(inputs1,inputs2, trained_model_path, w_en, c):
    # with tf.device('/gpu:1'):
    iqa1 = IQA(inputs = inputs1, trained_model_path = trained_model_path)
    iqa2 = IQA(inputs = inputs2, trained_model_path = trained_model_path)


    en1 = EN(inputs1)
    en2 = EN(inputs2)
    score1 = iqa1 + w_en * en1
    score2 = iqa2 + w_en * en2
    w1 = np.exp(score1 / c) / (np.exp(score1 / c) + np.exp(score2 / c))
    w2 = np.exp(score2 / c) / (np.exp(score1 / c) + np.exp(score2 / c))

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
    return (w1,w2)

class CONTENT_LOSS(nn.Module):
    def __init__(self, batch_size, device) -> None:
        super(CONTENT_LOSS, self).__init__()
        self.grad_loss = Fro_LOSS()
        self.per_loss = Per_LOSS()
        self.W1 = nn.Parameter(torch.randn(batch_size, 1, dtype=torch.float32))
        self.W2 = nn.Parameter(torch.randn(batch_size, 1, dtype=torch.float32))
        self.device = device

    def forward(self, img1, img2, f):
        pass
    
    
    def compute_fisher(self, imgset, iqa_model, samples=200):
        pass