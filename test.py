import numpy as np

from model.srgan import generator
from model import resolve_single
import cv2

gan_generator = generator()

gan_generator.load_weights('weight/pre_generator (1).h5')


def resolve_and_plot(lr_image_path, extension='.jpg'):
    file_name = lr_image_path.split('/')[-1].replace(extension, '')
    lr = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)
    channels = cv2.split(lr)
    if len(channels) == 4:
        data = cv2.merge([channels[0], channels[1], channels[2]])
        gan_sr = resolve_single(gan_generator, data)
    else:
        gan_sr = resolve_single(gan_generator, lr)

    # defoged_img = simplest_cb(np.array(gan_sr), 10)
    defoged_img = np.array(gan_sr)

    cv2.imwrite(f'./results/{file_name}_bicubic{extension}', defoged_img)


resolve_and_plot('input/test9.jpg', extension='.jpg')
