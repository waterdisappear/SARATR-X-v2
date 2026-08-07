import os
import pickle
from scipy.io import loadmat
import re
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from collections import defaultdict
import random
from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class FUSARShip(DatasetBase):
    dataset_dir = "FUSARShip"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_Li_SAR_ACD.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, 'images')
            # test_file = os.path.join(self.dataset_dir, 'images')
            trainval = self.read_data(trainval_file)
            # test = self.read_data(test_file)
            train, test_val = self.split_trainval(trainval, n_trn=20)
            # OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(test_val, num_shots=min(num_shots * 1, 10))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test_val, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def split_trainval(self, trainval, n_trn=20):
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) - n_trn)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val
    def read_data(self, image_dir):
        label_int = {'BulkCarrier': 0, 'ContainerShip': 1, 'Fishing': 2, 'GeneralCargo': 3, 'Tanker': 4}

        # label_name = {'BMP2': 'BMP2',
        #               'BTR70': 'BTR70',
        #               'T72': 'T72',
        #               'BTR60': 'BTR60',
        #               '2S1': '2S1',
        #               'BRDM2': 'BRDM2',
        #               'D7': 'D7',
        #               'T62': 'T62',
        #               'ZIL131': 'ZIL131',
        #               'ZSU234': 'ZSU234'}

        # label_name = {'A220': 'A220', 'A330': 'A330', 'ARJ21': 'ARJ21', 'Boeing737': 'Boeing737',
        #               'Boeing787': 'Boeing787'}

        items = []

        for root, dirs, files in os.walk(image_dir):
            files = sorted(files)
            for file in files:
                if os.path.splitext(file)[1] == '.tiff':
                    impath = os.path.join(root, file)
                    idx = re.split('[/\\\]', impath).index('FUSARShip')
                    label = label_int[re.split('[/\\\]', impath)[idx + 2]]
                    classname = re.split('[/\\\]', impath)[idx + 2]
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)
        return items
