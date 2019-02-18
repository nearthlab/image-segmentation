from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Add
from keras.layers import MaxPooling2D

############################################################
#  Feature Pyramid Network
############################################################

'''
    [Input - Output Sizes]
    input image size: (B, N, N, 3)
    => If backbone's stage outputs [C0, C1, C2, C3] have sizes [(B, N / 4, N / 4, *), (B, N / 8, N / 8, *), (B, N / 16, N / 16, *), (B, N / 32, N / 32, *)], respectively
    (*-values depend on backbone structure)
    then FPN's output tensors [P0, P1, P2, P3, P4] will be
    P0 = Conv1x1(C0) + Upsample(P1): (B, N / 4, N / 4, pyramid_size)
    P1 = Conv1x1(C1) + Upsample(P2): (B, N / 8, N / 8, pyramid_size)
    P2 = Conv1x1(C2) + Upsample(P3): (B, N / 16, N / 16, pyramid_size)
    P3 = Conv1x1(C3): (B, N / 32, N / 32, pyramid_size)
    P4 = MaxPool(P3): (B, N / 64, N / 64, pyramid_size)        
'''
# TODO: replace this with segmentation_models' FPN
def fpn_graph(input_tensors, pyramid_size = 256, interpolation = 'nearest'):
    num_inputs = len(input_tensors)
    output_tensors = [None] * num_inputs
    for i in range(num_inputs):
        # iterate through input_tensors backward:
        # j = num_inputs - 1, num_inputs - 2, ..., 1, 0
        j = num_inputs - i - 1
        if j == num_inputs - 1:
            output_tensors[j] = Conv2D(pyramid_size, (1, 1),
                                       name='fpn_c{}p{}'.format(j + 1, j + 1)
                                       )(input_tensors[j])
        else:
            output_tensors[j] = Add(name='fpn_p{}add'.format(j + 1))([
                UpSampling2D(
                    size=(2, 2), name='fpn_p{}upsample'.format(j + 2),
                    interpolation=interpolation
                )(output_tensors[j + 1]),
                Conv2D(
                    pyramid_size, (1, 1), name='fpn_c{}p{}'.format(j + 1, j + 1)
                )(input_tensors[j])
            ])

    for i in range(len(output_tensors)):
        output_tensors[i] = Conv2D(
            pyramid_size, (3, 3), padding='SAME', name='fpn_p{}'.format(i + 1)
        )(output_tensors[i])

    output_tensors.append(
        MaxPooling2D(
            pool_size=(1, 1), strides=2, name='fpn_p{}'.format(num_inputs + 1)
        )(output_tensors[-1])
    )

    return output_tensors

