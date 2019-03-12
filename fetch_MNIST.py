

import numpy as np
import struct
import matplotlib.pyplot as plt
train_images_idx3_ubyte_file = "../input_data/train-images.idx3-ubyte"
train_labels_idx1_ubyte_file = "../input_data/train-labels.idx1-ubyte"
test_images_idx3_ubyte_file = "../input_data/t10k-images.idx3-ubyte"
test_labels_idx1_ubyte_file = "../input_data/t10k-labels.idx1-ubyte"


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('%d,%d,%d*%d' % (magic_number, num_images, num_rows, num_cols))
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('%d' % (i + 1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('%d,%d' % (magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('%d' % (i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)




def run():
    train_images = load_test_images()
    train_labels = load_test_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()
    print(type(train_images), train_images.shape)
    print(type(train_labels), train_labels.shape)
    for i in range(10):
        print(train_labels[i])
        print(np.max(train_images), np.min(train_images))
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    print('done')

if __name__ == '__main__':
    run()


