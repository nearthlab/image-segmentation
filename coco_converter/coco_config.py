from collections import OrderedDict

# Define which colors match which categories in the images.
# For example:
# blade_id, nose_id, nacelle_id, body_id = [1, 2, 3, 4]
# category_ids = {
#     '(0, 255, 0)': blade_id,
#     '(0, 0, 255)': nose_id,
#     '(255, 0, 0)': nacelle_id,
#     '(255, 255, 0)': body_id
# }
# categories = [
#     OrderedDict(
#         [
#             ("supercategory", "windturbine"),
#             ("id", blade_id),
#             ("name", "blade")
#         ]
#     ),
#     OrderedDict(
#         [
#             ("supercategory", "windturbine"),
#             ("id", nose_id),
#             ("name", "nose")
#         ]
#     ),
#     OrderedDict(
#         [
#             ("supercategory", "windturbine"),
#             ("id", nacelle_id),
#             ("name", "nacelle")
#         ]
#     ),
#     OrderedDict(
#         [
#             ("supercategory", "windturbine"),
#             ("id", body_id),
#             ("name", "body")
#         ]
#     )
# ]

# blade_id = 1
# category_ids = {
#     '(0, 255, 0)': blade_id,
#     '(0, 254, 0)': blade_id,
#     '(0, 253, 0)': blade_id,
# }
# categories = [
#     OrderedDict(
#         [
#             ("supercategory", "windturbine"),
#             ("id", blade_id),
#             ("name", "blade")
#         ]
#     )
# ]

blade_id = 1
category_ids = {
    '(0, 255, 0)': blade_id,
    '(0, 254, 0)': blade_id,
    '(0, 253, 0)': blade_id,
    '(255, 255, 0)': blade_id
}
categories = [
    OrderedDict(
        [
            ("supercategory", "windturbine"),
            ("id", blade_id),
            ("name", "blade")
        ]
    )
]

# blade_id = 1
# nose_id = 2
# category_ids = {
#     '(0, 255, 0)': blade_id,
#     '(0, 254, 0)': blade_id,
#     '(0, 253, 0)': blade_id,
#     '(0, 0, 255)': nose_id
# }
# categories = [
#     OrderedDict(
#         [
#             ("supercategory", "windturbine"),
#             ("id", blade_id),
#             ("name", "blade")
#         ]
#     ),
#     OrderedDict(
#         [
#             ("supercategory", "windturbine"),
#             ("id", nose_id),
#             ("name", "nose")
#         ]
#     ),
# ]