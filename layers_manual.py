import math

import numpy as np
import tensorflow as tf

import layers_keras

BatchNormalization = layers_keras.BatchNormalization
Dropout = layers_keras.Dropout


class Conv2D(layers_keras.Conv2D):
    """
    Manually applies filters using the appropriate filter size and stride size
    """

    def call(self, inputs, training=False):
        ## If it's training, revert to layers implementation since this can be non-differentiable
        if training:
            return super().call(inputs, training)

        ## Otherwise, manually compute convolution at inference.
        ## Doesn't have to be differentiable. YAY!
        bn, h_in, w_in, c_in = inputs.shape  ## Batch #, height, width, # channels in input
        c_out = self.filters                 ## channels in output
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.strides                ## filter stride

        # Cleaning padding input.
        if self.padding == "SAME":
            if (h_in % sh == 0):
                ph = max(fh - sh, 0)
            else:
                ph = max(fh - (h_in % sh), 0)
            if (w_in % sw == 0):
                pw = max(fw - sw, 0)
            else:
                pw = max(fw - (w_in % sw), 0)

            oh = math.ceil(h_in / sh)
            ow  = math.ceil(w_in / sw)

            pad_top = ph // 2
            pad_bottom = ph - pad_top
            pad_left = pw // 2
            pad_right = pw - pad_left

            inputs = tf.pad(inputs, ([[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]]), "CONSTANT")

        elif self.padding == "VALID":
            ph, pw = 0, 0
            oh = math.ceil((h_in - fh + 1) / sh)
            ow  = math.ceil((w_in - fw + 1) / sw)
            
        else:
            raise AssertionError(f"Illegal padding type {self.padding}")

        outputs = np.zeros((bn, oh, ow, c_out), dtype="float32")

        for b in range(bn):
                for co in range(c_out):
                    for h in range(oh):
                        for w in range(ow):
                            target_matrix = inputs[b, h*sh : h * sh + fh, w*sw:w * sw + fw, :]
                            outputs[b, h, w, co] = tf.reduce_sum(target_matrix * self.kernel[:,:,:,co])
        return tf.convert_to_tensor(outputs, dtype=tf.float32)
