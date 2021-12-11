# the configuration is ViT-B_32
# the size of patch is (32,32)

import ml_collections

config = ml_collections.ConfigDict()
config.patches = ml_collections.ConfigDict()
config.patches.size = (32, 32)
config.hidden_size = 768
config.transformer = ml_collections.ConfigDict()
config.transformer.mlp_dim = 3072
config.transformer.num_heads = 12
config.transformer.num_layers = 12
config.transformer.attention_dropout_rate = 0.0
config.transformer.dropout_rate = 0.1
config.classifier = 'token'
config.representation_size = None
config.pretrain_dir = 'vit_checkpoint/ViT-B_32.npz'