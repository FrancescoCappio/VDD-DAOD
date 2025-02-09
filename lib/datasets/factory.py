# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.pascal_voc_cyclewater import pascal_voc_cyclewater
from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart
from datasets.sim10k import sim10k
from datasets.water import water
from datasets.clipart import clipart
from datasets.sim10k_cycle import sim10k_cycle
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape
from datasets.multi_weather import bdd
from datasets.voc_generic import voc_generic

import numpy as np
for split in ['train', 'trainval','val','test', 'train_s', 'test_s']:
    name = 'cityscape_{}'.format(split)
    __sets[name] = (lambda split=split : cityscape(split))
for split in ['foggy_test', 'foggy_train']:
    for data_split in ["", "10"]:
        name = 'foggy_cityscape{}_{}'.format(data_split,split)
        __sets[name] = (lambda split=split, data_split=data_split: cityscape(split,data_split=data_split))
"""
for split in ['train', 'trainval','val','test']:
    name = 'cityscape_car_{}'.format(split)
    __sets[name] = (lambda split=split : cityscape_car(split))
for split in ['train', 'trainval','test']:
    name = 'foggy_cityscape_{}'.format(split)
    __sets[name] = (lambda split=split : foggy_cityscape(split))
"""
for split in ['train','val']:
    name = 'sim10k_{}'.format(split)
    __sets[name] = (lambda split=split : sim10k(split))
for split in ['train', 'val']:
    name = 'sim10k_cycle_{}'.format(split)
    __sets[name] = (lambda split=split: sim10k_cycle(split))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_water_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc_water(split, year))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_cycleclipart_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart(split, year))
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_cyclewater_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc_cyclewater(split, year))
"""
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'clipart_{}'.format(split)
    __sets[name] = (lambda split=split : clipart(split,year))
for year in ['2007']:
    for split in ['train', 'test']:
        name = 'water_{}'.format(split)
        __sets[name] = (lambda split=split : water(split,year))
"""

for year in ['2007']:
    for split in ['train', 'test']:
        for ds in ['social_bikes', 'clipart', 'watercolor', 'comic']:
            for data_split in ['', '10']:
                name = '{}{}_{}'.format(ds,data_split,split)
                __sets[name] = (lambda split=split, ds=ds, data_split=data_split : voc_generic(ds, split, data_split))

for year in ['2007']:
    for split in ['train']:
        for weather in ['daytime_snow', 'daytime_foggy', 'night_rainy', 'daytime_sand', 'dusk_rainy', 'daytime_clear', 'night_clear']:
            name = 'bdd_{}'.format(weather)
            __sets[name] = (lambda weather=weather : bdd(split,year,weather))

for year in ['2007']:
    for split in ['train']:
        for weather in ['daytime_clear_in_daytime_foggy']:
            name = 'fft_{}'.format(weather)
            __sets[name] = (lambda weather=weather : bdd(split,year,weather))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
