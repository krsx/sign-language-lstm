import tensorflow as tf
import torch
# print(torch.cuda.is_available())

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())
