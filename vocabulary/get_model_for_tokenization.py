import tensorflow as tf
import tensorflow_hub as hub

def get_model(model_url, max_seq_length):
    with tf.device('/cpu:0'):
        labse_layer = hub.KerasLayer(model_url, trainable=False)

      # Define input.
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                             name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                     name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                          name="segment_ids")

      # LaBSE layer.
        pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

      # The embedding is l2 normalized.
        pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

      # Define model.
        labse_model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids],
        outputs=pooled_output)
    return labse_model, labse_layer