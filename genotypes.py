from collections import namedtuple

# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype = namedtuple('Genotype', 'ir vis decoder')
# Genotype2 = namedtuple('Genotype', 'ir vis decoder')

# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5'
# ]
PRIMITIVES = [
    # 'none',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    # 'res_conv_1x1',
    # 'res_conv_3x3',
    # 'res_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    # 'res_dil_conv_3x3',
    # 'res_dil_conv_5x5'
]
PRIMITIVES2 = [
    # 'none',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]

# NASNet = Genotype(
#   normal = [
#     ('sep_conv_5x5', 1),
#     ('sep_conv_3x3', 0),
#     ('sep_conv_5x5', 0),
#     ('sep_conv_3x3', 0),
#     ('avg_pool_3x3', 1),
#     ('skip_connect', 0),
#     ('avg_pool_3x3', 0),
#     ('avg_pool_3x3', 0),
#     ('sep_conv_3x3', 1),
#     ('skip_connect', 1),
#   ],
#   normal_concat = [2, 3, 4, 5, 6],
#   reduce = [
#     ('sep_conv_5x5', 1),
#     ('sep_conv_7x7', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('sep_conv_5x5', 0),
#     ('skip_connect', 3),
#     ('avg_pool_3x3', 2),
#     ('sep_conv_3x3', 2),
#     ('max_pool_3x3', 1),
#   ],
#   reduce_concat = [4, 5, 6],
# )
    
# AmoebaNet = Genotype(
#   normal = [
#     ('avg_pool_3x3', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_3x3', 0),
#     ('sep_conv_5x5', 2),
#     ('sep_conv_3x3', 0),
#     ('avg_pool_3x3', 3),
#     ('sep_conv_3x3', 1),
#     ('skip_connect', 1),
#     ('skip_connect', 0),
#     ('avg_pool_3x3', 1),
#     ],
#   normal_concat = [4, 5, 6],
#   reduce = [
#     ('avg_pool_3x3', 0),
#     ('sep_conv_3x3', 1),
#     ('max_pool_3x3', 0),
#     ('sep_conv_7x7', 2),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('max_pool_3x3', 0),
#     ('max_pool_3x3', 1),
#     ('conv_7x1_1x7', 0),
#     ('sep_conv_3x3', 5),
#   ],
#   reduce_concat = [3, 4, 6]
# )

# DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1),
#                             ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
#                     normal_concat=[2, 3, 4, 5], 
#                     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), 
#                             ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], 
#                     reduce_concat=[2, 3, 4, 5])
# DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), 
#                             ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], 
#                     normal_concat=[2, 3, 4, 5], 
#                     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), 
#                             ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], 
#                     reduce_concat=[2, 3, 4, 5])

# Test=Genotype(normal=[['max_pool_3x3', '0'], ['max_pool_3x3', '0'], ['max_pool_3x3', '0'], ['max_pool_3x3', '0']], normal_concat=[2, 3, 4, 5], reduce=[['max_pool_3x3', '1'], ['max_pool_3x3', '0'], ['max_pool_3x3', '2'], ['max_pool_3x3', '0'], ['max_pool_3x3', '2'], ['max_pool_3x3', '0'], ['max_pool_3x3', '2'], ['max_pool_3x3', '0']], reduce_concat=[2, 3, 4, 5])
# DARTS = DARTS_V2

# DARTS_stitch = Genotype(
# ir=[('res_dil_conv_3x3', 0), ('conv_1x1', 0), ('res_conv_3x3', 1), ('conv_1x1', 1), ('res_conv_3x3', 0), ('dil_conv_3x3', 3), ('res_conv_1x1', 0)], 
# vis=[('res_conv_3x3', 0), ('dil_conv_5x5', 0), ('res_dil_conv_3x3', 1), ('conv_3x3', 1), ('conv_1x1', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1)], 
# fuse=[('conv_3x3', 0), ('conv_1x1', 0), ('res_conv_1x1', 1), ('conv_3x3', 0), ('conv_1x1', 1), ('conv_3x3', 2), ('conv_1x1', 3), ('conv_3x3', 0), ('res_conv_1x1', 1)], 
# fuse2=[('res_conv_1x1', 0), ('conv_3x3', 1), ('res_conv_1x1', 0), ('conv_3x3', 0), ('conv_1x1', 2), ('res_conv_1x1', 0), ('res_conv_1x1', 1), ('res_conv_1x1', 2), ('conv_1x1', 1)], 
# cost=[('conv_5x5', 0), ('res_conv_5x5', 1), ('conv_5x5', 0), ('res_dil_conv_5x5', 1), ('res_conv_5x5', 2), ('dil_conv_3x3', 0), ('res_conv_1x1', 2)])
# DARTS_stitch =Genotype(
# ir=[('res_dil_conv_3x3', 0), ('conv_1x1', 0), ('res_dil_conv_3x3', 1), ('conv_1x1', 1), ('res_conv_3x3', 0), ('res_conv_1x1', 0), ('dil_conv_3x3', 3)], 
# vis=[('res_conv_3x3', 0), ('res_dil_conv_5x5', 0), ('res_dil_conv_3x3', 1), ('conv_3x3', 1), ('conv_1x1', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], 
# fuse=[('conv_3x3', 0), ('conv_1x1', 0), ('res_conv_1x1', 1), ('conv_3x3', 0), ('conv_1x1', 1), ('conv_3x3', 2), ('conv_1x1', 0), ('conv_3x3', 0), ('res_conv_1x1', 4)], 
# fuse2=[('res_conv_1x1', 0), ('conv_3x3', 1), ('res_conv_1x1', 0), ('conv_3x3', 0), ('conv_1x1', 2), ('res_conv_1x1', 1), ('res_conv_1x1', 0), ('conv_1x1', 1), ('conv_5x5', 3)], 
# cost=[('conv_5x5', 0), ('res_conv_3x3', 1), ('conv_5x5', 0), ('res_dil_conv_5x5', 1), ('res_conv_5x5', 2), ('dil_conv_3x3', 0), ('res_conv_1x1', 2)])
# DARTS_stitch =Genotype(
# ir=[('res_dil_conv_3x3', 0), ('conv_1x1', 0), ('res_dil_conv_3x3', 1), ('conv_1x1', 1), ('res_conv_3x3', 0), ('res_conv_1x1', 0), ('dil_conv_3x3', 3)], 
# vis=[('res_conv_3x3', 0), ('res_dil_conv_5x5', 0), ('res_dil_conv_3x3', 1), ('conv_3x3', 1), ('conv_1x1', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], 
# fuse=[('conv_3x3', 0), ('conv_1x1', 0), ('dil_conv_3x3', 1), ('conv_3x3', 0), ('conv_1x1', 1), ('conv_3x3', 2), ('conv_1x1', 0), ('conv_3x3', 0), ('dil_conv_3x3', 4)], 
# fuse2=[('dil_conv_3x3', 0), ('conv_3x3', 1), ('dil_conv_3x3', 0), ('conv_3x3', 0), ('conv_1x1', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('conv_1x1', 1), ('conv_5x5', 3)], 
# cost=[('conv_5x5', 0), ('res_conv_3x3', 1), ('conv_5x5', 0), ('res_dil_conv_5x5', 1), ('res_conv_5x5', 2), ('dil_conv_3x3', 0), ('res_conv_1x1', 2)])
# DARTS_fussion = Genotype(
#     ir=[('conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('conv_1x1', 1), ('dil_conv_5x5', 2), ('conv_1x1', 2), ('dil_conv_3x3', 3)], 
#     vis=[('dil_conv_5x5', 0), ('conv_3x3', 1), ('conv_5x5', 0), ('conv_1x1', 2), ('dil_conv_3x3', 1), ('conv_1x1', 2), ('conv_1x1', 0)], 
#     decoder=[('conv_1x1', 0), ('dil_conv_5x5', 0), ('conv_3x3', 1), ('conv_1x1', 2), ('dil_conv_3x3', 1), ('conv_1x1', 1), ('conv_1x1', 0)])
DARTS_fusion = Genotype(
                        ir=[('conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('conv_1x1', 1), ('dil_conv_5x5', 2), ('conv_1x1', 2), ('conv_3x3', 1)], 
                        vis=[('dil_conv_5x5', 0), ('conv_3x3', 1), ('conv_5x5', 0), ('conv_1x1', 2), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_1x1', 0)], 
                        decoder=[('conv_1x1', 0), ('dil_conv_5x5', 0), ('conv_3x3', 1), ('conv_1x1', 2), ('dil_conv_3x3', 1), ('conv_1x1', 1), ('conv_1x1', 0)]
                        )
DARTS_fusion_nonas = Genotype(
                        ir=[('conv_5x5', 0), ('conv_5x5', 0), ('conv_5x5', 1), ('conv_5x5', 1), ('conv_5x5', 2), ('conv_5x5', 2), ('conv_5x5', 1)], 
                        vis=[('conv_5x5', 0), ('conv_5x5', 1), ('conv_5x5', 0), ('conv_5x5', 2), ('conv_5x5', 1), ('conv_5x5', 2), ('conv_5x5', 0)], 
                        decoder=[('conv_5x5', 0), ('conv_5x5', 0), ('conv_3x3', 1), ('conv_5x5', 2), ('conv_5x5', 1), ('conv_5x5', 1), ('conv_5x5', 0)]
                        )
