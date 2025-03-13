# Dataset Splitting

The `TatmDataset` class includes functionality for creating simple index based train and validation scripts 
that can be used to separate the dataset into training and validation sets. The functionality as implemented 
allows for users to either specify a number of indices or a percentage of the dataset to be used for validation.
The split will be deterministic based on the index in the full dataset and will not change between runs or if the 
same dataset is loaded multiple times.

## Example loading a dataset and splitting it into training and validation sets

```python
from tatm import get_dataset

dataset = get_dataset("my_data", context_length=512, val_split_size=0.1)
print(len(dataset))
# 1000

```

The proceeding code will load the dataset and tell the dataset object to prepare to split the dataset into a training
and validation set where the validation set will be 10% of the full dataset. However if we call `len` on the dataset
at this point we will see that the dataset is still the full dataset.

If we want to use the training set for the split we have two possible approaches. The first is to call the `set_split`
method on the dataset object and pass in the string "train" as the argument. The second is to pass "train" as the
split argument when initializing the dataset.

```python
dataset.set_split("train")
print(len(dataset))
# 900

train_dataset = get_dataset("my_data", context_length=512, val_split_size=0.1, split="train")
print(len(train_dataset))
# 900
```

We can also use the same approach to get the validation set.

```python
dataset.set_split("val")
print(len(dataset))
# 100

val_dataset = get_dataset("my_data", context_length=512, val_split_size=150, split="val") # we can also pass in a number of items to use for the validation set
print(len(val_dataset))
# 150
```

Note that we can use the `set_split` method to switch between the training and validation sets at any time. If we want to operate on the full dataset we can call `set_split(None)` or pass `None` as the split argument when initializing the dataset. If we have loaded a dataset without defining a split size, we can still create a split by calling the `create_split` method and passing in the desired split size. This will create a new split based on the current dataset and the specified split size. Note that this has to be done prior to calling `set_split` or passing in a split argument when initializing the dataset. 

```python
dataset = get_dataset("my_data", context_length=512)
print(len(dataset))
# 1000
dataset.create_split(0.1) # create a split with 10% of the dataset reserved for validation
print(len(dataset))
# 1000
dataset.set_split("train")
print(len(dataset))
# 900
dataset.set_split("val")
print(len(dataset))
# 100
dataset.set_split(None) # set the split to None to use the full dataset
print(len(dataset))
# 1000
```



When we have the splits created, the indices used to return items in the dataset will be remapped to only return items from the split that we are using. Note that 
this also means in the case of the validation split, indices will be remapped so that the first index in the validation split can be returned by calling `dataset[0]`.

```python
dataset = get_dataset("my_data", context_length=512, val_split_size=0.2)
dataset.set_split(None)
val_dataset = get_dataset("my_data", context_length=512, val_split_size=0.2, split="val")
print(dataset[800] == val_dataset[0])
# True
```

With these features in place, we can easily create training and validation sets for our dataset and use them in training and evaluation loops as drop in replacements for the full dataset.