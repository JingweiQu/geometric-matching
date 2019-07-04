import os
import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt


def show_image(image_name, title, index):
    image = io.imread(image_name)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=2)
    ax = plt.subplot(2, 3, index)
    ax.set_title(title)
    ax.imshow(image)


image_id = range(0, 800, 200)
test_dataset = 'TSS'

for id in image_id:
    source_name = os.path.join('geometric_matching/demo_results/finetune/with_box', test_dataset, str(id) + '_source_image.jpg')
    target_name = os.path.join('geometric_matching/demo_results/finetune/with_box', test_dataset, str(id) + '_target_image.jpg')
    warped_1_name = os.path.join('geometric_matching/demo_results/finetune/with_box', test_dataset, str(id) + '_warped_image_aff_tps.jpg')
    warped_2_name = os.path.join('geometric_matching/demo_results/20190623/with_box', test_dataset, str(id) + '_warped_image_aff_tps.jpg')
    warped_3_name = os.path.join('geometric_matching/demo_results/20190613/with_box', test_dataset, str(id) + '_warped_image_aff_tps.jpg')
    warped_4_name = os.path.join('geometric_matching/demo_results/weak/', test_dataset, str(id) + '_warped_image_affine_tps.jpg')

    show_image(source_name, 'source', 1)
    show_image(target_name, 'target', 4)
    show_image(warped_1_name, 'warped_finetune', 2)
    show_image(warped_2_name, 'warped_20190623', 3)
    show_image(warped_3_name, 'warped_20190613', 5)
    show_image(warped_4_name, 'warped_weak', 6)
    plt.show()