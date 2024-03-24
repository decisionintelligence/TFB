# -*- coding: utf-8 -*-
class Singleton(type):
    """
    Used to construct singleton classes through the method of meta classes
    """

    _instance_dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance_dict:
            cls._instance_dict[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance_dict[cls]
