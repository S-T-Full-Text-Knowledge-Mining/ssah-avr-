import h5py
import numpy as np
from scipy import io

def loading_data_zyf(path):
        print ('******************************************************')
        print ('dataset:{0}'.format(path))
        print ('******************************************************')

        file_Label = io.loadmat('./data/LAll/mirflickr25k-lall.mat')
        file_image = h5py.File('./data/IAll/mirflickr25k-iall.mat')
        file_text = io.loadmat('./data/YAll/mirflickr25k-yall.mat')
        # print(type(file_image))
        # file_Label = np.load('./data1/l.npy')
        # file_image = np.load('./data1/i.npy')
        # file_text = np.load('./data1/t.npy')
        images = file_image['IAll'][:4033].transpose(0,3,2,1)
        labels = file_Label['LAll'][:4033]
        tags = file_text['YAll'][:4033]
        # images = file_image[:500]
        # labels = file_Label[:500]
        # tags = file_text[:500]
        # file_image.close()
        print(images.shape)
        print(type(images))
        print(labels.shape)
        print(type(labels))
        print(tags.shape)
        print(type(tags))

        return images, tags, labels

def loading_data(path):
	print ('******************************************************')
	print ('dataset:{0}'.format(path))
	print ('******************************************************')

	file = h5py.File(path)
	images = file['images'][:30].transpose(0,3,2,1)
	labels = file['LAll'][:30].transpose(1,0)
	tags = file['YAll'][:30].transpose(1,0)
	file.close()

	return images, tags, labels


def split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE):

	X = {}
	index_all = np.random.permutation(QUERY_SIZE+DATABASE_SIZE)
	ind_Q = index_all[0:QUERY_SIZE]
	ind_T = index_all[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE]
	ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

	X['query'] = images[ind_Q, :, :, :]
	X['train'] = images[ind_T, :, :, :]
	X['retrieval'] = images[ind_R, :, :, :]

	Y = {}
	Y['query'] = tags[ind_Q, :]
	Y['train'] = tags[ind_T, :]
	Y['retrieval'] = tags[ind_R, :]

	L = {}
	L['query'] = labels[ind_Q, :]
	L['train'] = labels[ind_T, :]
	L['retrieval'] = labels[ind_R, :]
	return X, Y, L
