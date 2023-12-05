import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())
# print(tf.sysconfig.get_build_info()['cuda_version'])
