# %%
import numpy as np 
from scipy import misc
np.random.seed(1)

# filename structure
path = 'D:/OneDrive/Nam_4_1/ImageProcessing/PCAandRecognization/YALE/unpadded/'
ids = range(1, 16)  # 15 persons
states = ['centerlight', 'glasses', 'happy', 'leftlight',
          'noglasses', 'normal', 'rightlight','sad',
          'sleepy', 'surprised', 'wink']
prefix = 'subject'
surfix = '.pgm'

# D:/OneDrive/Nam_4_1/ImageProcessing/PCAandRecognization/YALE/unpadded
# %%
# data dimension
h = 116
w = 98
D = h * w
N = len(states) * 15
K = 100
D
# %%
# collect all data
import imageio
X = np.zeros((D, N))
cnt = 0
for person_id in range(1, 16):
	for state in states:
		fn = path + prefix + str(person_id).zfill(2) + '.' +state+surfix
		X[:, cnt] = imageio.imread(fn).reshape(D)
		cnt += 1

# %%
# doing PCA, note that each row is a datapoint
from sklearn.decomposition import  PCA
pca = PCA(n_components=K)
pca.fit(X.T)

# %%
# projection matrix
U = pca.components_.T
# %%
U.shape
X.shape
# %%
import matplotlib.pyplot as plt 
for i in range(U.shape[1]):
	plt.axis('off')
	f1 = plt.imshow(U[:, i].reshape(h, w), interpolation='nearest')
	f1.axes.get_xaxis().set_visible(False)
	f1.axes.get_yaxis().set_visible(False)
	#     f2 = plt.imshow(, interpolation='nearest' )
	plt.gray()
	fn = 'eigenface' + str(i).zfill(2) + '.png'
	plt.savefig(fn, bbox_inches='tight', pad_inches=0)
	# plt.show()

# %%
# see reconstruction of the first 6 persons
for person_id in range(1, 7):
	for state in ['centerlight']:
		fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
		im = imageio.imread(fn)
		plt.axis('off')
		#         plt.imshow(im, interpolation='nearest' )
		f1 = plt.imshow(im, interpolation='nearest')
		f1.axes.get_xaxis().set_visible(False)
		f1.axes.get_yaxis().set_visible(False)
		plt.gray()
		fn = 'ori' + str(person_id).zfill(2) + '.png'
		plt.savefig(fn, bbox_inches='tight', pad_inches=0)
		plt.show()
		# reshape and subtract mean, don't forget 
		x = im.reshape(D, 1) - pca.mean_.reshape(D, 1)
		# encode
		z = U.T.dot(x)
		#decode
		x_tilde = U.dot(z) + pca.mean_.reshape(D, 1)

		# reshape to orginal dim
		im_tilde = x_tilde.reshape(116, 98)
		plt.axis('off')
		#         plt.imshow(im_tilde, interpolation='nearest' )
		f1 = plt.imshow(im_tilde, interpolation='nearest')
		f1.axes.get_xaxis().set_visible(False)
		f1.axes.get_yaxis().set_visible(False)
		plt.gray()
		fn = 'res' + str(person_id).zfill(2) + '.png'
		plt.savefig(fn, bbox_inches='tight', pad_inches=0)
		plt.show()

# %%
cnt = 0 
for person_id in [10]:
    for ii, state in enumerate(states):
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        im = imageio.imread(fn)
        f1 = plt.imshow(im, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)

        fn = 'ex' + str(ii).zfill(2) +  '.png'
        plt.axis('off')
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
         
        plt.show()
#         cnt += 1
# %%