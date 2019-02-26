from collections import OrderedDict

category_ids = {
    #'(0, 0, 0)': unlabeled
    '(4, 4, 4)': 1,
    '(5, 5, 5)': 2,
    '(6, 6, 6)': 3,
    '(7, 7, 7)': 4,
    '(8, 8, 8)': 5,
    '(9, 9, 9)': 6,
    '(10, 10, 10)': 7,
    '(11, 11, 11)': 8,
    '(12, 12, 12)': 9,
    '(13, 13, 13)': 10,
    '(14, 14, 14)': 11,
    '(15, 15, 15)': 12,
    '(16, 16, 16)': 13,
    '(17, 17, 17)': 14,
    '(18, 18, 18)': 15,
    '(19, 19, 19)': 16,
    '(20, 20, 20)': 17,
    '(21, 21, 21)': 18,
    '(22, 22, 22)': 19,
    '(23, 23, 23)': 20,
    '(24, 24, 24)': 21,
    '(25, 25, 25)': 22,
    '(26, 26, 26)': 23,
    '(27, 27, 27)': 24,
    '(28, 28, 28)': 25,
    '(29, 29, 29)': 26,
    '(30, 30, 30)': 27,
    '(31, 31, 31)': 28,
    '(32, 32, 32)': 29,
    '(33, 33, 33)': 30
}

categories = [
    OrderedDict(
        [
            ("supercategory", "void"),
            ("id", 1),
            ("name", 'static')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "void"),
            ("id", 2),
            ("name", 'dynamic')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "void"),
            ("id", 3),
            ("name", 'ground')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "flat"),
            ("id", 4),
            ("name", 'road')
        ]
    ),OrderedDict(
        [
            ("supercategory", "flat"),
            ("id", 5),
            ("name", 'sidewalk')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "flat"),
            ("id", 6),
            ("name", 'parking')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "flat"),
            ("id", 7),
            ("name", 'rail track')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "contruction"),
            ("id", 8),
            ("name", 'building')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "contruction"),
            ("id", 9),
            ("name", 'wall')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "contruction"),
            ("id", 10),
            ("name", 'fence')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "contruction"),
            ("id", 11),
            ("name", 'guard rail')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "contruction"),
            ("id", 12),
            ("name", 'bridge')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "contruction"),
            ("id", 13),
            ("name", 'tunnel')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "object"),
            ("id", 14),
            ("name", 'pole')
        ]
    ),OrderedDict(
        [
            ("supercategory", "object"),
            ("id", 15),
            ("name", 'pole group')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "object"),
            ("id", 16),
            ("name", 'traffic light')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "object"),
            ("id", 17),
            ("name", 'traffic sign')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "nature"),
            ("id", 18),
            ("name", 'vegetation')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "nature"),
            ("id", 19),
            ("name", 'terrain')
        ]
    ),OrderedDict(
        [
            ("supercategory", "sky"),
            ("id", 20),
            ("name", 'sky')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "human"),
            ("id", 21),
            ("name", 'person')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "human"),
            ("id", 22),
            ("name", 'rider')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "vehicle"),
            ("id", 23),
            ("name", 'car')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "vehicle"),
            ("id", 24),
            ("name", 'truck')
        ]
    ),OrderedDict(
        [
            ("supercategory", "vehicle"),
            ("id", 25),
            ("name", 'bus')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "vehicle"),
            ("id", 26),
            ("name", 'caravan')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "vehicle"),
            ("id", 27),
            ("name", 'trailer')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "vehicle"),
            ("id", 28),
            ("name", 'train')
        ]
    ),
    OrderedDict(
        [
            ("supercategory", "vehicle"),
            ("id", 29),
            ("name", 'motorcycle')
        ]
    ),OrderedDict(
        [
            ("supercategory", "vehicle"),
            ("id", 30),
            ("name", 'bicycle')
        ]
    ),
]
