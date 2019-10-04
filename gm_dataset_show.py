import os
import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from translate import Translator

def show_image(image_name, title, index):
    image = io.imread(image_name)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=2)
    ax = plt.subplot(4, 8, index)
    ax.set_title(title)
    ax.imshow(image)

test_dataset = 'TSS'
csv_file = os.path.join('geometric_matching/testing_data/', test_dataset, 'test_' + test_dataset + '.csv')
image_path = os.path.join('geometric_matching/testing_data/', test_dataset)
dataframe = pd.read_csv(csv_file)  # Read images data
categories = dataframe.iloc[:, 2].values.astype('float')  # Get image category
category_names = ['Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor',
                               'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus',
                               'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone',
                               'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile',
                               'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly',
                               'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo',
                               'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill',
                               'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo',
                               'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah',
                               'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon',
                               'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner',
                               'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish',
                               'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella',
                               'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
# translator = Translator(to_lang='chinese')
category = None
if category is not None:
    cat_idx = np.nonzero(categories == category)[0]
    categories = categories[cat_idx]
    dataframe = dataframe.iloc[cat_idx, :]
img_A_names = dataframe.iloc[:, 0]  # Get source image & target image name
img_B_names = dataframe.iloc[:, 1]


for i in range(0, int(len(dataframe)/2)-1):
    source_name = os.path.join(image_path, img_A_names[i*2])
    target_name = os.path.join(image_path, img_B_names[i*2])
    # print(source_name, target_name)

    show_image(source_name, 'source', (i % 16) * 2 + 1)
    show_image(target_name, 'target', (i % 16) * 2 + 2)
    if (i+1) % 16 == 0:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
# for category in range(98, 102):
#     cat_idx = np.nonzero(categories == category)[0]
#     img_A_names = dataframe.iloc[cat_idx, 0]  # Get source image & target image name
#     img_B_names = dataframe.iloc[cat_idx, 1]
#
#     j = 1
#     for i in cat_idx:
#         source_name = os.path.join(image_path, img_A_names[i])
#         target_name = os.path.join(image_path, img_B_names[i])
#         # print(source_name, target_name)
#
#         k = i + j - cat_idx[0]
#         show_image(source_name, 'source', k)
#         show_image(target_name, 'target', k+1)
#         j += 1
#
#     print(translator.translate(category_names[category-1]))
#     plt.suptitle(str(category) + '-' + category_names[category-1])
#     mng = plt.get_current_fig_manager()
#     mng.window.showMaximized()
#     plt.show()