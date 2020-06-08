"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
#from data.base_dataset import BaseDataset

from .BaseDataset import BaseDataset
from .InputExample import InputExample
from .ParallelSentencesDataset import ParallelSentencesDataset
#from .SentenceLabelDataset import SentenceLabelDataset
from .SentencesDataset import SentencesDataset


def find_dataset_using_name(dataset_type):
    """Import the module "data/[dataset_type]Dataset.py".

    In the file, the class called DatasetTypeDataset() will
    be instantiated. It has to be a subclass of BaseDataset (non nel mio),
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_type + "Dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_type = dataset_type.replace('_', '') + 'Dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_type.lower() and issubclass(cls, BaseDataset) :
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a class with class name that matches %s in lowercase." % (dataset_filename, target_dataset_type))

    return dataset


def get_option_setter(dataset_type):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_type)
    return dataset_class.modify_commandline_options


def create_dataset(opt, model):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt, model)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, model):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt, model)
        self.dataset.load_data()
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
