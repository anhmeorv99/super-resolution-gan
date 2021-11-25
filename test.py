import numpy as np

from model.srgan import generator
from model import resolve_single

import cv2

gan_generator = generator()

gan_generator.load_weights('weight/pre_generator (1).h5')


def resolve_and_plot(lr_image_path, file_name):
    lr = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)

    gan_sr = resolve_single(gan_generator, lr)

    cv2.imwrite(f'{file_name}_sr.jpg', np.array(gan_sr))


resolve_and_plot('biensovang.jpg', 'biensovang')
