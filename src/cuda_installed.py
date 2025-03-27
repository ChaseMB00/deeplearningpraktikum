import tensorflow as tf
import timeit

print("Tensorflow v", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print(gpus)
# tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print(
            '\n\nThis error most likely means that this notebook is not '
            'configured to use a GPU.  Change this in Notebook Settings via the '
            'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n'
            )
    raise SystemError('GPU device not found')

height = 200
width = 200
batch_size = 128


# filters = [[[1, 1], [1, 1],[0,1],[1,1]]]


def cpu():
    with tf.device('/cpu:0'):
        random_image_cpu = tf.random.normal((batch_size, height, width, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        # tf.nn.conv2d(random_image_cpu, filters, strides=[1], padding="VALID")
        return tf.math.reduce_sum(net_cpu)


def gpu():
    with tf.device('/device:GPU:0'):
        random_image_gpu = tf.random.normal((batch_size, height, width, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)


# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
print("--- warmup --------------------------")
print("Start CPU")
cpu()
print("Start GPU")
gpu()
print("--- warmup finished -----------------")
number_of_runs = 30

# Run the op several times.
print(f'Time (s) to convolve 32x7x7x3 filter over random images {height}x{width}x3 (batch x height x width x '
      f'channel). Sum of {number_of_runs} runs.'
      )
cpu_time = timeit.timeit('cpu()', number=number_of_runs, setup="from __main__ import cpu")
print('CPU (s): {:,.4f}'.format(cpu_time))
gpu_time = timeit.timeit('gpu()', number=number_of_runs, setup="from __main__ import gpu")
print('GPU (s): {:,.4f}'.format(gpu_time))
print(f"GPU speedup over CPU: {cpu_time / gpu_time:.1f}x")

# with tensorflow-metal
# Start GPU
# --- warmup finished -----------------
# Time (s) to convolve 32x7x7x3 filter over random images 200x200x3 (batch x height x width x channel). Sum of 30 runs.
# CPU (s): 7.6102
# GPU (s): 0.6176
# GPU speedup over CPU: 12.3x