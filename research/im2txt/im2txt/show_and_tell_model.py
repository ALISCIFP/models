# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
tf.test.is_gpu_available()

import numpy as np
import os

from im2txt.ops import image_embedding
from im2txt.ops import image_processing
from im2txt.ops import inputs as input_ops


class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # doc2vec
    # self.input_vecs = None
    # self.target_vecs = None
    # self.mask_vec = None
    self.vectors = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    self.semantic_features = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    # doc2vec

    # print("Path at terminal when executing this file:")
    # print(os.getcwd() + "\n")

    # print("about to load vecs")
    # self.vectors = np.loadtxt('./train_corpus_vectors.txt')
    # print("loaded vecs")


  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
      vectors = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions_and_vectors = []
      # images_and_captions = []
      # vectors = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption, vector = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name,
            vector_feature="image/vector")
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions_and_vectors.append([image, caption, vector])
        # images_and_captions.append([image, caption])      
        # vectors.append(vector)

      # print("")
      # print("image_feature_name: " + str(self.config.image_feature_name))
      # print("caption_feature_name: " + str(self.config.caption_feature_name))
      # print("")

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      # images, input_seqs, target_seqs, input_mask, input_vecs, target_vecs, mask_vec = input_ops.batch_with_dynamic_pad(images_and_captions_and_vectors,
      images, input_seqs, target_seqs, input_mask, vectors = input_ops.batch_with_dynamic_pad(images_and_captions_and_vectors,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity)

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask
    # self.input_vecs = input_vecs
    # self.target_vecs = target_vecs
    # self.mask_vec = mask_vec
    self.vectors = vectors

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    inception_output = image_embedding.inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")
    
    # print("image_embedding: " + str((image_embeddings).shape))

    # convert dimensions into (512,)
    semantic_features = tf.contrib.layers.flatten(image_embeddings)
    semantic_features = tf.layers.dense(inputs=semantic_features, units=512)

    # self.image_embeddings = image_embeddings
    self.semantic_features = semantic_features

    ####################################################################################

    # flatten_1 = tf.contrib.layers.flatten(image_embeddings)
    fc_1 = tf.layers.dense(inputs=image_embeddings, units=512)
    # flatten_2 = tf.contrib.layers.flatten(semantic_features)
    fc_2 = tf.layers.dense(inputs=semantic_features, units=512)

    image_embeddings = fc_1 + fc_2

    #####################################################################################
    

    self.image_embeddings = image_embeddings


  def coattention(self, avg_features, semantic_features, h_sent):
    
    embed_size = 512
    hidden_size = 512  # size of h_sent, semantic_features
    visual_size = 512 # size of avg_features

    # input has to have size = visual_size
    W_v = tf.layers.dense(inputs=avg_features, units=visual_size)
    # input has to have size = hidden_size
    W_v_h = tf.layers.dense(inputs=tf.squeeze(h_sent, [1]), units=visual_size)

    alpha_v_input = tf.math.tanh(tf.math.add(W_v, W_v_h))
    alpha_v_dense = tf.layers.dense(inputs=alpha_v_input, units=visual_size)
    alpha_v = tf.nn.softmax(alpha_v_dense)
    v_att = tf.math.multiply(alpha_v, avg_features)

    # input must be of size = hidden_size
    W_a_h = tf.layers.dense(inputs=h_sent, units=hidden_size)
    W_a = tf.layers.dense(inputs=semantic_features, units=hidden_size)
    alpha_a_input = tf.math.tanh(tf.math.add(W_a_h, W_a))
    alpha_a_dense = tf.layers.dense(inputs=alpha_a_input, units=hidden_size)
    alpha_a = tf.nn.softmax(alpha_a_dense)
    a_att_multiply = tf.math.multiply(alpha_a, semantic_features)
    a_att = tf.math.reduce_sum(a_att_multiply, 1)

    ctx_cat = tf.concat([v_att, a_att], 1)
    ctx = tf.layers.dense(inputs=ctx_cat, units=embed_size)

    return ctx, alpha_v, alpha_a


  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

    self.seq_embeddings = seq_embeddings

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
      _, initial_state = lstm_cell(self.image_embeddings, zero_state)

      print("")
      print("initial_state[0].shape: " + str(initial_state[0].shape))
      print("initial_state[1].shape: " + str(initial_state[1].shape))
      print("")

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        tf.concat(axis=1, values=initial_state, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
        tf.concat(axis=1, values=state_tuple, name="state")
      else:
        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = tf.reduce_sum(self.input_mask, 1)
        print("self.seq_embeddings.shape: " + str((self.seq_embeddings).shape))
        # print("sequence_length: " + str(sequence_length))
        # print("self.image_embeddings,shape: " + str((self.image_embeddings).shape))
        lstm_outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=self.seq_embeddings,
                                            sequence_length=sequence_length,
                                            initial_state=initial_state,
                                            dtype=tf.float32,
                                            scope=lstm_scope)

        # print("")
        # print("lstm_outputs.shape: " + str(lstm_outputs.shape))
        # print("state.shape: " + str(state[0].shape))
        # lstm_outputs.shape: (32, ?, 512)
        # state.shape: (32, 512)
        # print("")



        # self.seq_embeddings.shape: (32, ?, 512)

        # for i in range((self.seq_embeddings).shape[0]):
        #   caption = self.seq_embeddings[i]
        #   word_count = (caption.shape)[0] 

        # print("")
        # print("type: " + str(type((self.seq_embeddings).get_shape[1])))
        # print("self.seq_embeddings.shape[1]: " + str((self.seq_embeddings).get_shape[1]))
        # print("")
       
        
        # for j in range((tf.shape(self.seq_embeddings)[1]).eval()):
        #   if j == 0:
        #     _, state_output_tuple = lstm_cell((self.seq_embeddings)[j], initial_state[0])
        #   else:
        #     _, state_output_tuple = lstm_cell((self.seq_embeddings)[j], ctx)

        #   hidden_state = state_output_tuple[0]
        #   output = state_output_tuple[1]
            
        #   ctx, alpha_v, alpha_a = coattention(self.image_embeddings, self.semantic_features, hidden_state)            
          



    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)

    if self.mode == "inference":
      tf.nn.softmax(logits, name="softmax")
    else:
      targets = tf.reshape(self.target_seqs, [-1])
      weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                              logits=logits)
      batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                          tf.reduce_sum(weights),
                          name="batch_loss")
      tf.losses.add_loss(batch_loss)

      # L1 loss between CNN vector and doc2vec vector
      alpha = 0.01
      l1_loss = tf.losses.absolute_difference(self.vectors, self.semantic_features)
      tf.losses.add_loss(alpha*l1_loss)

      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/l1_loss", l1_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
