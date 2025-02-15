import os
import pandas as pd
import scipy.io as sio
from .imagenet_prompts import imagenet_classes, cifar10_classes, cifar100_classes
from .fewshot_datasets import fewshot_datasets
from .imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask


root = '/data2/chentao/data/check_data'

class_names = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'classes.txt'), sep=' ', names=['class_id', 'target'])
bird200_classes = [name.split(".")[1].replace('_', ' ') for name in class_names.target]

car196_classes = sio.loadmat(os.path.join(root, 'stanford_cars/devkit', 'cars_meta.mat'), squeeze_me=True)["class_names"].tolist()

food101_classes = ['Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare', 'Beet salad', 'Beignets', 'Bibimbap', 'Bread pudding', 'Breakfast burrito', 'Bruschetta', 'Caesar salad', 'Cannoli', 'Caprese salad', 'Carrot cake', 'Ceviche', 'Cheesecake', 'Cheese plate', 'Chicken curry', 'Chicken quesadilla', 'Chicken wings', 'Chocolate cake', 'Chocolate mousse', 'Churros', 'Clam chowder', 'Club sandwich', 'Crab cakes', 'Creme brulee', 'Croque madame', 'Cup cakes', 'Deviled eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs benedict', 'Escargots', 'Falafel', 'Filet mignon', 'Fish and chips', 'Foie gras', 'French fries', 'French onion soup', 'French toast', 'Fried calamari', 'Fried rice', 'Frozen yogurt', 'Garlic bread', 'Gnocchi', 'Greek salad', 'Grilled cheese sandwich', 'Grilled salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot and sour soup', 'Hot dog', 'Huevos rancheros', 'Hummus', 'Ice cream', 'Lasagna', 'Lobster bisque', 'Lobster roll sandwich', 'Macaroni and cheese', 'Macarons', 'Miso soup', 'Mussels', 'Nachos', 'Omelette', 'Onion rings', 'Oysters', 'Pad thai', 'Paella', 'Pancakes', 'Panna cotta', 'Peking duck', 'Pho', 'Pizza', 'Pork chop', 'Poutine', 'Prime rib', 'Pulled pork sandwich', 'Ramen', 'Ravioli', 'Red velvet cake', 'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed salad', 'Shrimp and grits', 'Spaghetti bolognese', 'Spaghetti carbonara', 'Spring rolls', 'Steak', 'Strawberry shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles']

pet_path = os.path.join(root, 'oxford-iiit-pet/annotations', 'test.txt')
pet_image_ids = []
pet_labels = []
with open(pet_path) as file:
    for line in file:
        image_id, label, *_ = line.strip().split()
        pet_image_ids.append(image_id)
        pet_labels.append(int(label) - 1)

pet37_classes = [
    " ".join(part.title() for part in raw_cls.split("_"))
    for raw_cls, _ in sorted(
        {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(pet_image_ids, pet_labels)},
        key=lambda image_id_and_label: image_id_and_label[1],
    )
]


def get_classnames(set_id):
    dataset_class_map = {
        'bird200': bird200_classes,
        'car196': car196_classes,
        'food101': food101_classes,
        'pet37': pet37_classes,
        'CIFAR-100': cifar100_classes, 'CIFAR-100-C': cifar100_classes, 'CIFAR-100-C-OOD': cifar100_classes,
        'CIFAR-10': cifar10_classes, 'CIFAR-10-C': cifar10_classes, 'CIFAR-10-C-OOD': cifar10_classes,
    }

    if set_id in fewshot_datasets:
        classnames = eval(f"{set_id.lower()}_classes")
    elif set_id in dataset_class_map:
        classnames = dataset_class_map[set_id]
    elif set_id in ['A', 'R', 'K', 'V', 'I', 'ImageNet-C']:
        classnames_all = imagenet_classes
        if set_id in ['A', 'R', 'V']:
            label_mask = eval(f"imagenet_{set_id.lower()}_mask")
            if set_id == 'R':
                classnames = [classnames_all[i] for i, m in enumerate(label_mask) if m]
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    else:
        raise ValueError(f"Unknown dataset ID: {set_id}")

    return classnames
