# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Temporal Fusion Transformer Model.

Contains the full TFT architecture and associated components. Defines functions
for training, evaluation and prediction using simple Pandas Dataframe inputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import json
import os
import shutil

import data_formatters.base
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow as tf

# Layer definitions.
concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda

# Default input types.
InputTypes = data_formatters.base.InputTypes


# Layer utility functions.
def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
  """Returns simple Keras linear layer.

  Args:
    size: Output size
    activation: Activation function to apply if required
    use_time_distributed: Whether to apply layer across time
    use_bias: Whether bias should be included in layer
  """
  linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
  if use_time_distributed:
    linear = tf.keras.layers.TimeDistributed(linear)
  return linear


def apply_mlp(inputs,
              hidden_size,
              output_size,
              output_activation=None,
              hidden_activation='tanh',
              use_time_distributed=False):
  """Applies simple feed-forward network to an input.

  Args:
    inputs: MLP inputs
    hidden_size: Hidden state size
    output_size: Output size of MLP
    output_activation: Activation function to apply on output
    hidden_activation: Activation function to apply on input
    use_time_distributed: Whether to apply across time

  Returns:
    Tensor for MLP outputs.
  """
  if use_time_distributed:
    hidden = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_size, activation=hidden_activation))(
            inputs)
    return tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_size, activation=output_activation))(
            hidden)
  else:
    hidden = tf.keras.layers.Dense(
        hidden_size, activation=hidden_activation)(
            inputs)
    return tf.keras.layers.Dense(
        output_size, activation=output_activation)(
            hidden)


def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):
  """Applies a Gated Linear Unit (GLU) to an input.

  Args:
    x: Input to gating layer
    hidden_layer_size: Dimension of GLU
    dropout_rate: Dropout rate to apply if any
    use_time_distributed: Whether to apply across time
    activation: Activation function to apply to the linear feature transform if
      necessary

  Returns:
    Tuple of tensors for: (GLU output, gate)
  """

  if dropout_rate is not None:
    x = tf.keras.layers.Dropout(dropout_rate)(x)

  if use_time_distributed:
    activation_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_layer_size, activation=activation))(
            x)
    gated_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(
            x)
  else:
    activation_layer = tf.keras.layers.Dense(
        hidden_layer_size, activation=activation)(
            x)
    gated_layer = tf.keras.layers.Dense(
        hidden_layer_size, activation='sigmoid')(
            x)

  return tf.keras.layers.Multiply()([activation_layer,
                                     gated_layer]), gated_layer


def add_and_norm(x_list):
  """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
  tmp = Add()(x_list)
  tmp = LayerNorm()(tmp)
  return tmp


def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
  """Applies the gated residual network (GRN) as defined in paper.

  Args:
    x: Network inputs
    hidden_layer_size: Internal state size
    output_size: Size of output layer
    dropout_rate: Dropout rate if dropout is applied
    use_time_distributed: Whether to apply network across time dimension
    additional_context: Additional context vector to use if relevant
    return_gate: Whether to return GLU gate for diagnostic purposes

  Returns:
    Tuple of tensors for: (GRN output, GLU gate)
  """

  # Setup skip connection
  if output_size is None:
    output_size = hidden_layer_size
    skip = x
  else:
    linear = Dense(output_size)
    if use_time_distributed:
      linear = tf.keras.layers.TimeDistributed(linear)
    skip = linear(x)

  # Apply feedforward network
  hidden = linear_layer(
      hidden_layer_size,
      activation=None,
      use_time_distributed=use_time_distributed)(
          x)
  if additional_context is not None:
    hidden = hidden + linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed,
        use_bias=False)(
            additional_context)
  hidden = tf.keras.layers.Activation('elu')(hidden)
  hidden = linear_layer(
      hidden_layer_size,
      activation=None,
      use_time_distributed=use_time_distributed)(
          hidden)

  gating_layer, gate = apply_gating_layer(
      hidden,
      output_size,
      dropout_rate=dropout_rate,
      use_time_distributed=use_time_distributed,
      activation=None)

  if return_gate:
    return add_and_norm([skip, gating_layer]), gate
  else:
    return add_and_norm([skip, gating_layer])def linear_layer(size, activation=None, use_time_distributed=False, use_bias=True):
    """Crea una capa lineal simple en Keras (equivalente a nn.Linear en PyTorch).
    
    En PyTorch sería:
    layer = nn.Linear(in_features, size, bias=use_bias)
    if activation:
        layer = nn.Sequential(layer, activation)
    
    Args:
        size: Tamaño de salida de la capa
        activation: Función de activación (similar a PyTorch)
        use_time_distributed: Si True, aplica la capa a cada paso temporal
                            (en PyTorch esto se hace manualmente con un bucle o reshape)
        use_bias: Si incluir bias (igual que en PyTorch)
    """
    # Dense es equivalente a nn.Linear en PyTorch
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    
    # TimeDistributed aplica la capa a cada timestep
    # En PyTorch esto se haría:
    # for t in range(sequence_length):
    #     output[:, t, :] = linear(input[:, t, :])
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear

def apply_mlp(inputs, hidden_size, output_size, output_activation=None,
              hidden_activation='tanh', use_time_distributed=False):
    """Aplica una red feed-forward simple (MLP).
    
    En PyTorch sería algo como:
    class MLP(nn.Module):
        def __init__(self):
            self.hidden = nn.Linear(input_size, hidden_size)
            self.output = nn.Linear(hidden_size, output_size)
            self.hidden_act = nn.Tanh()  # o la activación especificada
            
    Args:
        inputs: Entrada al MLP
        hidden_size: Tamaño de la capa oculta
        output_size: Tamaño de la salida
        output_activation: Activación final
        hidden_activation: Activación de la capa oculta
        use_time_distributed: Si aplicar a través del tiempo
    """
    if use_time_distributed:
        # En PyTorch esto requeriría un bucle sobre timesteps o reshape
        hidden = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_size, activation=hidden_activation))(inputs)
        return tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(output_size, activation=output_activation))(hidden)
    else:
        # Esto es más similar a PyTorch, una capa tras otra
        hidden = tf.keras.layers.Dense(hidden_size, activation=hidden_activation)(inputs)
        return tf.keras.layers.Dense(output_size, activation=output_activation)(hidden)

def apply_gating_layer(x, hidden_layer_size, dropout_rate=None,
                      use_time_distributed=True, activation=None):
    """Aplica una Gated Linear Unit (GLU) - Unidad Lineal con Compuerta.
    
    En PyTorch sería:
    class GLU(nn.Module):
        def __init__(self):
            self.linear = nn.Linear(input_size, hidden_layer_size)
            self.gate = nn.Linear(input_size, hidden_layer_size)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
    
    Args:
        x: Input a la capa
        hidden_layer_size: Dimensión de la GLU
        dropout_rate: Tasa de dropout (igual que en PyTorch)
        use_time_distributed: Si aplicar en dimensión temporal
        activation: Activación opcional para la transformación lineal
    """
    # Aplicar dropout si se especifica
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        # Capa de activación
        activation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation=activation))(x)
        # Capa de compuerta (gate)
        gated_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(x)
    else:
        activation_layer = tf.keras.layers.Dense(hidden_layer_size, activation=activation)(x)
        gated_layer = tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid')(x)

    # Multiplicación elemento a elemento (en PyTorch: torch.mul(activation_layer, gated_layer))
    return tf.keras.layers.Multiply()([activation_layer, gated_layer]), gated_layer

def add_and_norm(x_list):
    """Aplica conexión residual (skip connection) seguida de normalización.
    
    En PyTorch:
    class AddAndNorm(nn.Module):
        def __init__(self):
            self.norm = nn.LayerNorm(normalized_shape)
        
        def forward(self, x_list):
            return self.norm(torch.add(x_list[0], x_list[1]))
    
    Args:
        x_list: Lista de tensores para sumar en la conexión skip
    """
    # Suma (en PyTorch: torch.add)
    tmp = Add()(x_list)
    # Normalización (en PyTorch: nn.LayerNorm)
    tmp = LayerNorm()(tmp)
    return tmp

def gated_residual_network(x, hidden_layer_size, output_size=None, dropout_rate=None,
                          use_time_distributed=True, additional_context=None, return_gate=False):
    """Implementa la Gated Residual Network (GRN) del paper TFT.
    
    Esta es una parte clave de la arquitectura TFT que combina:
    - Conexiones residuales
    - Gating mechanism
    - Layer normalization
    
    En PyTorch sería una clase que hereda de nn.Module combinando los componentes anteriores.
    
    Args:
        x: Entrada a la red
        hidden_layer_size: Tamaño del estado interno
        output_size: Tamaño de la capa de salida (si None, usa hidden_layer_size)
        dropout_rate: Dropout si se aplica
        use_time_distributed: Si aplicar a través del tiempo
        additional_context: Vector de contexto adicional opcional
        return_gate: Si devolver la compuerta GLU para diagnóstico
    """
    # Configurar skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        # En PyTorch: linear = nn.Linear(input_size, output_size)
        linear = Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    # Red feedforward
    hidden = linear_layer(hidden_layer_size, activation=None,
                         use_time_distributed=use_time_distributed)(x)
    
    # Incorporar contexto adicional si existe
    if additional_context is not None:
        hidden = hidden + linear_layer(hidden_layer_size, activation=None,
                                     use_time_distributed=use_time_distributed,
                                     use_bias=False)(additional_context)
    
    # Activación ELU (en PyTorch: nn.ELU())
    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(hidden_layer_size, activation=None,
                         use_time_distributed=use_time_distributed)(hidden)

    # Aplicar capa GLU
    gating_layer, gate = apply_gating_layer(hidden, output_size,
                                          dropout_rate=dropout_rate,
                                          use_time_distributed=use_time_distributed,
                                          activation=None)

    # Retornar resultado con o sin la compuerta
    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])


# Attention Components.
def get_decoder_mask(self_attn_inputs):
    """Returns causal mask to apply for self-attention layer.

    Args:
        self_attn_inputs: Inputs to self-attention layer to determine mask shape
    """
    # Obtiene la longitud de la secuencia de las entradas
    len_s = tf.shape(input=self_attn_inputs)[1]
    # Obtiene el tamaño del batch
    bs = tf.shape(input=self_attn_inputs)[:1]
    # Crea una máscara causal (superior triangular), usada para que el modelo no vea información futura
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class ScaledDotProductAttention():
    """Defines scaled dot product attention layer.

    Attributes:
        dropout: Tasa de abandono (dropout) a aplicar
        activation: Función de normalización para la atención (por defecto softmax)
    """

    def __init__(self, attn_dropout=0.0):
        self.dropout = Dropout(attn_dropout)  # Capa de dropout
        self.activation = Activation('softmax')  # Activación softmax por defecto

    def __call__(self, q, k, v, mask):
        """Aplica la atención por producto punto escalado.

        Args:
            q: Consultas (queries)
            k: Claves (keys)
            v: Valores (values)
            mask: Máscara a aplicar si es necesario (evita que se vea información futura)

        Returns:
            Tuple de (salida de la capa, pesos de la atención)
        """
        # Calcula el factor de escala (sqrt de la dimensión de las claves)
        temper = tf.sqrt(tf.cast(tf.shape(input=k)[-1], dtype='float32'))
        
        # Aplica el producto punto entre las consultas y las claves
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # Dimensión=(batch, q, k)
        
        # Si existe una máscara, la añade a los puntajes de atención
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)  # Configura valores muy negativos
            attn = Add()([attn, mmask])
        
        # Normaliza los puntajes con softmax
        attn = self.activation(attn)
        # Aplica la capa de dropout
        attn = self.dropout(attn)
        
        # Calcula la salida final aplicando el producto punto entre la atención y los valores
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class InterpretableMultiHeadAttention():
    """Define la capa de atención multi-cabeza interpretable.

    Attributes:
        n_head: Número de cabezas
        d_k: Dimensionalidad de clave/consulta por cabeza
        d_v: Dimensionalidad de valores
        dropout: Tasa de abandono (dropout)
        qs_layers: Lista de consultas por cabeza
        ks_layers: Lista de claves por cabeza
        vs_layers: Lista de valores por cabeza
        attention: Capa de atención por producto punto escalado
        w_o: Matriz de pesos de salida para proyectar el estado interno al tamaño original del TFT
    """

    def __init__(self, n_head, d_model, dropout):
        """Inicializa la capa de atención multi-cabeza interpretable.

        Args:
            n_head: Número de cabezas
            d_model: Dimensionalidad del estado TFT
            dropout: Tasa de abandono
        """
        self.n_head = n_head
        # Calcula la dimensionalidad de cada cabeza (dividiendo el tamaño de d_model entre n_head)
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = []  # Lista de capas para consultas
        self.ks_layers = []  # Lista de capas para claves
        self.vs_layers = []  # Lista de capas para valores

        # Crea una capa de valores que será compartida entre las cabezas
        vs_layer = Dense(d_v, use_bias=False)

        # Crea las capas para consultas, claves y valores para cada cabeza
        for _ in range(n_head):
            self.qs_layers.append(Dense(d_k, use_bias=False))
            self.ks_layers.append(Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)  # Usamos la misma capa de valores

        # Crea una capa de atención por producto punto escalado
        self.attention = ScaledDotProductAttention()
        # Crea una capa densa para la proyección de salida
        self.w_o = Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """Aplica la atención multi-cabeza interpretable.

        Args:
            q: Consulta de forma=(?, T, d_model)
            k: Clave de forma=(?, T, d_model)
            v: Valores de forma=(?, T, d_model)
            mask: Máscara (si es necesaria) de forma=(?, T, T)

        Returns:
            Tuple de (salida de la capa, pesos de atención)
        """
        n_head = self.n_head

        heads = []  # Lista para almacenar las salidas de cada cabeza
        attns = []  # Lista para almacenar los pesos de atención de cada cabeza
        for i in range(n_head):
            # Obtiene las consultas, claves y valores para la cabeza i
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            # Aplica la atención para la cabeza i
            head, attn = self.attention(qs, ks, vs, mask)

            # Aplica el dropout a la salida de la cabeza
            head_dropout = Dropout(self.dropout)(head)
            heads.append(head_dropout)  # Almacena la salida de la cabeza
            attns.append(attn)  # Almacena los pesos de atención

        # Apila las salidas de las cabezas si hay más de una cabeza
        head = K.stack(heads) if n_head > 1 else heads[0]
        attn = K.stack(attns)

        # Calcula la media de las salidas de las cabezas si hay más de una cabeza
        outputs = K.mean(head, axis=0) if n_head > 1 else head
        # Aplica la proyección final de salida
        outputs = self.w_o(outputs)
        # Aplica el dropout en la salida final
        outputs = Dropout(self.dropout)(outputs)

        return outputs, attn


class TFTDataCache(object):
    """Cache de datos para el TFT."""

    _data_cache = {}

    @classmethod
    def update(cls, data, key):
        """Actualiza los datos en la caché.

        Args:
            data: Fuente de los datos para actualizar
            key: Clave para la ubicación en el diccionario
        """
        cls._data_cache[key] = data

    @classmethod
    def get(cls, key):
        """Devuelve los datos almacenados en la clave dada."""
        return cls._data_cache[key].copy()

    @classmethod
    def contains(cls, key):
        """Devuelve un valor booleano indicando si la clave está presente en la caché."""
        return key in cls._data_cache


# TFT model definitions.
class TemporalFusionTransformer(object):
  """Defines Temporal Fusion Transformer.

  Attributes:
    name: Name of model
    time_steps: Total number of input time steps per forecast date (i.e. Width
      of Temporal fusion decoder N)
    input_size: Total number of inputs
    output_size: Total number of outputs
    category_counts: Number of categories per categorical variable
    n_multiprocessing_workers: Number of workers to use for parallel
      computations
    column_definition: List of tuples of (string, DataType, InputType) that
      define each column
    quantiles: Quantiles to forecast for TFT
    use_cudnn: Whether to use Keras CuDNNLSTM or standard LSTM layers
    hidden_layer_size: Internal state size of TFT
    dropout_rate: Dropout discard rate
    max_gradient_norm: Maximum norm for gradient clipping
    learning_rate: Initial learning rate of ADAM optimizer
    minibatch_size: Size of minibatches for training
    num_epochs: Maximum number of epochs for training
    early_stopping_patience: Maximum number of iterations of non-improvement
      before early stopping kicks in
    num_encoder_steps: Size of LSTM encoder -- i.e. number of past time steps
      before forecast date to use
    num_stacks: Number of self-attention layers to apply (default is 1 for basic
      TFT)
    num_heads: Number of heads for interpretable mulit-head attention
    model: Keras model for TFT
  """

  def __init__(self, raw_params, use_cudnn=False):
        """Construye el TFT con los parámetros proporcionados.

        Args:
            raw_params: Parámetros para definir el TFT.
            use_cudnn: Si se usa LSTM optimizado para GPU con CuDNN.
        """

        self.name = self.__class__.__name__  # Asigna el nombre del modelo a partir de la clase

        params = dict(raw_params)  # Copia los parámetros para su uso local

        # Parámetros de datos
        self.time_steps = int(params['total_time_steps'])  # Total de pasos de tiempo de entrada
        self.input_size = int(params['input_size'])  # Número de entradas al modelo
        self.output_size = int(params['output_size'])  # Número de salidas del modelo
        self.category_counts = json.loads(str(params['category_counts']))  # Número de categorías por variable categórica
        self.n_multiprocessing_workers = int(params['multiprocessing_workers'])  # Número de trabajadores para cálculos paralelos

        # Índices relevantes para el modelo TFT
        self._input_obs_loc = json.loads(str(params['input_obs_loc']))  # Índices para las observaciones de entrada
        self._static_input_loc = json.loads(str(params['static_input_loc']))  # Índices para las entradas estáticas
        self._known_regular_input_idx = json.loads(str(params['known_regular_inputs']))  # Índices para entradas regulares conocidas
        self._known_categorical_input_idx = json.loads(str(params['known_categorical_inputs']))  # Índices para entradas categóricas conocidas

        self.column_definition = params['column_definition']  # Definición de columnas para las entradas

        # Parámetros de la red
        self.quantiles = [0.1, 0.5, 0.9]  # Cuantiles que se pronosticarán
        self.use_cudnn = use_cudnn  # Si usar optimización CuDNN para LSTM
        self.hidden_layer_size = int(params['hidden_layer_size'])  # Tamaño de la capa oculta
        self.dropout_rate = float(params['dropout_rate'])  # Tasa de deserción para evitar sobreajuste
        self.max_gradient_norm = float(params['max_gradient_norm'])  # Máximo norm para el recorte de gradientes
        self.learning_rate = float(params['learning_rate'])  # Tasa de aprendizaje del optimizador
        self.minibatch_size = int(params['minibatch_size'])  # Tamaño de los lotes mini para entrenamiento
        self.num_epochs = int(params['num_epochs'])  # Número máximo de épocas para entrenamiento
        self.early_stopping_patience = int(params['early_stopping_patience'])  # Paciencia para detención anticipada

        self.num_encoder_steps = int(params['num_encoder_steps'])  # Tamaño del codificador LSTM
        self.num_stacks = int(params['stack_size'])  # Número de capas de autoatención
        self.num_heads = int(params['num_heads'])  # Número de cabezas de atención múltiple

        # Configuración para serialización
        self._temp_folder = os.path.join(params['model_folder'], 'tmp')  # Carpeta temporal para almacenamiento intermedio
        self.reset_temp_folder()  # Restablece la carpeta temporal

        # Componentes adicionales para almacenar nodos de Tensorflow para cálculos de atención
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        print('*** {} params ***'.format(self.name))  # Imprime el nombre del modelo
        for k in params:  # Imprime todos los parámetros configurados
            print('# {} = {}'.format(k, params[k]))

 

  def get_tft_embeddings(self, all_inputs):
    """Transforms raw inputs to embeddings.

    Applies linear transformation onto continuous variables and uses embeddings
    for categorical variables.

    Args:
      all_inputs: Inputs to transform

    Returns:
      Tensors for transformed inputs.
    """

    time_steps = self.time_steps

    # Sanity checks
    for i in self._known_regular_input_idx:
      if i in self._input_obs_loc:
        raise ValueError('Observation cannot be known a priori!')
    for i in self._input_obs_loc:
      if i in self._static_input_loc:
        raise ValueError('Observation cannot be static!')

    if all_inputs.get_shape().as_list()[-1] != self.input_size:
      raise ValueError(
          'Illegal number of inputs! Inputs observed={}, expected={}'.format(
              all_inputs.get_shape().as_list()[-1], self.input_size))

    num_categorical_variables = len(self.category_counts)
    num_regular_variables = self.input_size - num_categorical_variables

    embedding_sizes = [
        self.hidden_layer_size for i, size in enumerate(self.category_counts)
    ]

    embeddings = []
    for i in range(num_categorical_variables):

      embedding = tf.keras.Sequential([
          tf.keras.layers.InputLayer([time_steps]),
          tf.keras.layers.Embedding(
              self.category_counts[i],
              embedding_sizes[i],
              input_length=time_steps,
              dtype=tf.float32)
      ])
      embeddings.append(embedding)

    regular_inputs, categorical_inputs \
        = all_inputs[:, :, :num_regular_variables], \
          all_inputs[:, :, num_regular_variables:]

    embedded_inputs = [
        embeddings[i](categorical_inputs[Ellipsis, i])
        for i in range(num_categorical_variables)
    ]

    # Static inputs
    if self._static_input_loc:
      static_inputs = [tf.keras.layers.Dense(self.hidden_layer_size)(
          regular_inputs[:, 0, i:i + 1]) for i in range(num_regular_variables)
                       if i in self._static_input_loc] \
          + [embedded_inputs[i][:, 0, :]
             for i in range(num_categorical_variables)
             if i + num_regular_variables in self._static_input_loc]
      static_inputs = tf.keras.backend.stack(static_inputs, axis=1)

    else:
      static_inputs = None

    def convert_real_to_embedding(x):
      """Applies linear transformation for time-varying inputs."""
      return tf.keras.layers.TimeDistributed(
          tf.keras.layers.Dense(self.hidden_layer_size))(
              x)

    # Targets
    obs_inputs = tf.keras.backend.stack([
        convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1])
        for i in self._input_obs_loc
    ],
                                        axis=-1)

    # Observed (a prioir unknown) inputs
    wired_embeddings = []
    for i in range(num_categorical_variables):
      if i not in self._known_categorical_input_idx \
        and  i + num_regular_variables  not in self._input_obs_loc:
        e = embeddings[i](categorical_inputs[:, :, i])
        wired_embeddings.append(e)

    unknown_inputs = []
    for i in range(regular_inputs.shape[-1]):
      if i not in self._known_regular_input_idx \
          and i not in self._input_obs_loc:
        e = convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1])
        unknown_inputs.append(e)

    if unknown_inputs + wired_embeddings:
      unknown_inputs = tf.keras.backend.stack(
          unknown_inputs + wired_embeddings, axis=-1)
    else:
      unknown_inputs = None

    # A priori known inputs
    known_regular_inputs = [
        convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1])
        for i in self._known_regular_input_idx
        if i not in self._static_input_loc
    ]
    known_categorical_inputs = [
        embedded_inputs[i]
        for i in self._known_categorical_input_idx
        if i + num_regular_variables not in self._static_input_loc
    ]

    known_combined_layer = tf.keras.backend.stack(
        known_regular_inputs + known_categorical_inputs, axis=-1)

    return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

  def _get_single_col_by_type(self, input_type):
    """Returns name of single column for input type."""

    return utils.get_single_col_by_input_type(input_type,
                                              self.column_definition)

  def training_data_cached(self):
    """Returns boolean indicating if training data has been cached."""

    return TFTDataCache.contains('train') and TFTDataCache.contains('valid')

  def cache_batched_data(self, data, cache_key, num_samples=-1):
    """Batches and caches data once for using during training.

    Args:
      data: Data to batch and cache
      cache_key: Key used for cache
      num_samples: Maximum number of samples to extract (-1 to use all data)
    """

    if num_samples > 0:
      TFTDataCache.update(
          self._batch_sampled_data(data, max_samples=num_samples), cache_key)
    else:
      TFTDataCache.update(self._batch_data(data), cache_key)

    print('Cached data "{}" updated'.format(cache_key))

  def _batch_sampled_data(self, data, max_samples):
    """Samples segments into a compatible format.

    Args:
      data: Sources data to sample and batch
      max_samples: Maximum number of samples in batch

    Returns:
      Dictionary of batched data with the maximum samples specified.
    """

    if max_samples < 1:
      raise ValueError(
          'Illegal number of samples specified! samples={}'.format(max_samples))

    id_col = self._get_single_col_by_type(InputTypes.ID)
    time_col = self._get_single_col_by_type(InputTypes.TIME)

    data.sort_values(by=[id_col, time_col], inplace=True)

    print('Getting valid sampling locations.')
    valid_sampling_locations = []
    split_data_map = {}
    for identifier, df in data.groupby(id_col):
      print('Getting locations for {}'.format(identifier))
      num_entries = len(df)
      if num_entries >= self.time_steps:
        valid_sampling_locations += [
            (identifier, self.time_steps + i)
            for i in range(num_entries - self.time_steps + 1)
        ]
      split_data_map[identifier] = df

    inputs = np.zeros((max_samples, self.time_steps, self.input_size))
    outputs = np.zeros((max_samples, self.time_steps, self.output_size))
    time = np.empty((max_samples, self.time_steps, 1), dtype=object)
    identifiers = np.empty((max_samples, self.time_steps, 1), dtype=object)

    if max_samples > 0 and len(valid_sampling_locations) > max_samples:
      print('Extracting {} samples...'.format(max_samples))
      ranges = [
          valid_sampling_locations[i] for i in np.random.choice(
              len(valid_sampling_locations), max_samples, replace=False)
      ]
    else:
      print('Max samples={} exceeds # available segments={}'.format(
          max_samples, len(valid_sampling_locations)))
      ranges = valid_sampling_locations

    id_col = self._get_single_col_by_type(InputTypes.ID)
    time_col = self._get_single_col_by_type(InputTypes.TIME)
    target_col = self._get_single_col_by_type(InputTypes.TARGET)
    input_cols = [
        tup[0]
        for tup in self.column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    for i, tup in enumerate(ranges):
      if (i + 1 % 1000) == 0:
        print(i + 1, 'of', max_samples, 'samples done...')
      identifier, start_idx = tup
      sliced = split_data_map[identifier].iloc[start_idx -
                                               self.time_steps:start_idx]
      inputs[i, :, :] = sliced[input_cols]
      outputs[i, :, :] = sliced[[target_col]]
      time[i, :, 0] = sliced[time_col]
      identifiers[i, :, 0] = sliced[id_col]

    sampled_data = {
        'inputs': inputs,
        'outputs': outputs[:, self.num_encoder_steps:, :],
        'active_entries': np.ones_like(outputs[:, self.num_encoder_steps:, :]),
        'time': time,
        'identifier': identifiers
    }

    return sampled_data

  def _batch_data(self, data):
    """Batches data for training.

    Converts raw dataframe from a 2-D tabular format to a batched 3-D array
    to feed into Keras model.

    Args:
      data: DataFrame to batch

    Returns:
      Batched Numpy array with shape=(?, self.time_steps, self.input_size)
    """

    # Functions.
    def _batch_single_entity(input_data):
      time_steps = len(input_data)
      lags = self.time_steps
      x = input_data.values
      if time_steps >= lags:
        return np.stack(
            [x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)

      else:
        return None

    id_col = self._get_single_col_by_type(InputTypes.ID)
    time_col = self._get_single_col_by_type(InputTypes.TIME)
    target_col = self._get_single_col_by_type(InputTypes.TARGET)
    input_cols = [
        tup[0]
        for tup in self.column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    data_map = {}
    for _, sliced in data.groupby(id_col):

      col_mappings = {
          'identifier': [id_col],
          'time': [time_col],
          'outputs': [target_col],
          'inputs': input_cols
      }

      for k in col_mappings:
        cols = col_mappings[k]
        arr = _batch_single_entity(sliced[cols].copy())

        if k not in data_map:
          data_map[k] = [arr]
        else:
          data_map[k].append(arr)

    # Combine all data
    for k in data_map:
      data_map[k] = np.concatenate(data_map[k], axis=0)

    # Shorten target so we only get decoder steps
    data_map['outputs'] = data_map['outputs'][:, self.num_encoder_steps:, :]

    active_entries = np.ones_like(data_map['outputs'])
    if 'active_entries' not in data_map:
      data_map['active_entries'] = active_entries
    else:
      data_map['active_entries'].append(active_entries)

    return data_map

  def _get_active_locations(self, x):
    """Formats sample weights for Keras training."""
    return (np.sum(x, axis=-1) > 0.0) * 1.0

  def _build_base_graph(self):
    """Returns graph defining layers of the TFT."""

    # Size definitions.
    time_steps = self.time_steps
    combined_input_size = self.input_size
    encoder_steps = self.num_encoder_steps

    # Inputs.
    all_inputs = tf.keras.layers.Input(
        shape=(
            time_steps,
            combined_input_size,
        ))

    unknown_inputs, known_combined_layer, obs_inputs, static_inputs \
        = self.get_tft_embeddings(all_inputs)

    # Isolate known and observed historical inputs.
    if unknown_inputs is not None:
      historical_inputs = concat([
          unknown_inputs[:, :encoder_steps, :],
          known_combined_layer[:, :encoder_steps, :],
          obs_inputs[:, :encoder_steps, :]
      ],
                                 axis=-1)
    else:
      historical_inputs = concat([
          known_combined_layer[:, :encoder_steps, :],
          obs_inputs[:, :encoder_steps, :]
      ],
                                 axis=-1)

    # Isolate only known future inputs.
    future_inputs = known_combined_layer[:, encoder_steps:, :]

    def static_combine_and_mask(embedding):
      """Applies variable selection network to static inputs.

      Args:
        embedding: Transformed static inputs

      Returns:
        Tensor output for variable selection network
      """

      # Add temporal features
      _, num_static, _ = embedding.get_shape().as_list()

      flatten = tf.keras.layers.Flatten()(embedding)

      # Nonlinear transformation with gated residual network.
      mlp_outputs = gated_residual_network(
          flatten,
          self.hidden_layer_size,
          output_size=num_static,
          dropout_rate=self.dropout_rate,
          use_time_distributed=False,
          additional_context=None)

      sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
      sparse_weights = K.expand_dims(sparse_weights, axis=-1)

      trans_emb_list = []
      for i in range(num_static):
        e = gated_residual_network(
            embedding[:, i:i + 1, :],
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        trans_emb_list.append(e)

      transformed_embedding = concat(trans_emb_list, axis=1)

      combined = tf.keras.layers.Multiply()(
          [sparse_weights, transformed_embedding])

      static_vec = K.sum(combined, axis=1)

      return static_vec, sparse_weights

    static_encoder, static_weights = static_combine_and_mask(static_inputs)

    static_context_variable_selection = gated_residual_network(
        static_encoder,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        use_time_distributed=False)
    static_context_enrichment = gated_residual_network(
        static_encoder,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        use_time_distributed=False)
    static_context_state_h = gated_residual_network(
        static_encoder,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        use_time_distributed=False)
    static_context_state_c = gated_residual_network(
        static_encoder,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        use_time_distributed=False)

    def lstm_combine_and_mask(embedding):
      """Apply temporal variable selection networks.

      Args:
        embedding: Transformed inputs.

      Returns:
        Processed tensor outputs.
      """

      # Add temporal features
      _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()

      flatten = K.reshape(embedding,
                          [-1, time_steps, embedding_dim * num_inputs])

      expanded_static_context = K.expand_dims(
          static_context_variable_selection, axis=1)

      # Variable selection weights
      mlp_outputs, static_gate = gated_residual_network(
          flatten,
          self.hidden_layer_size,
          output_size=num_inputs,
          dropout_rate=self.dropout_rate,
          use_time_distributed=True,
          additional_context=expanded_static_context,
          return_gate=True)

      sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
      sparse_weights = tf.expand_dims(sparse_weights, axis=2)

      # Non-linear Processing & weight application
      trans_emb_list = []
      for i in range(num_inputs):
        grn_output = gated_residual_network(
            embedding[Ellipsis, i],
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True)
        trans_emb_list.append(grn_output)

      transformed_embedding = stack(trans_emb_list, axis=-1)

      combined = tf.keras.layers.Multiply()(
          [sparse_weights, transformed_embedding])
      temporal_ctx = K.sum(combined, axis=-1)

      return temporal_ctx, sparse_weights, static_gate

    historical_features, historical_flags, _ = lstm_combine_and_mask(
        historical_inputs)
    future_features, future_flags, _ = lstm_combine_and_mask(future_inputs)

    # LSTM layer
    def get_lstm(return_state):
      """Returns LSTM cell initialized with default parameters."""
      if self.use_cudnn:
        lstm = tf.compat.v1.keras.layers.CuDNNLSTM(
            self.hidden_layer_size,
            return_sequences=True,
            return_state=return_state,
            stateful=False,
        )
      else:
        lstm = tf.keras.layers.LSTM(
            self.hidden_layer_size,
            return_sequences=True,
            return_state=return_state,
            stateful=False,
            # Additional params to ensure LSTM matches CuDNN, See TF 2.0 :
            # (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True)
      return lstm

    history_lstm, state_h, state_c \
        = get_lstm(return_state=True)(historical_features,
                                      initial_state=[static_context_state_h,
                                                     static_context_state_c])

    future_lstm = get_lstm(return_state=False)(
        future_features, initial_state=[state_h, state_c])

    lstm_layer = concat([history_lstm, future_lstm], axis=1)

    # Apply gated skip connection
    input_embeddings = concat([historical_features, future_features], axis=1)

    lstm_layer, _ = apply_gating_layer(
        lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None)
    temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

    # Static enrichment layers
    expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)
    enriched, _ = gated_residual_network(
        temporal_feature_layer,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        use_time_distributed=True,
        additional_context=expanded_static_context,
        return_gate=True)

    # Decoder self attention
    self_attn_layer = InterpretableMultiHeadAttention(
        self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate)

    mask = get_decoder_mask(enriched)
    x, self_att \
        = self_attn_layer(enriched, enriched, enriched,
                          mask=mask)

    x, _ = apply_gating_layer(
        x,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        activation=None)
    x = add_and_norm([x, enriched])

    # Nonlinear processing on outputs
    decoder = gated_residual_network(
        x,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        use_time_distributed=True)

    # Final skip connection
    decoder, _ = apply_gating_layer(
        decoder, self.hidden_layer_size, activation=None)
    transformer_layer = add_and_norm([decoder, temporal_feature_layer])

    # Attention components for explainability
    attention_components = {
        # Temporal attention weights
        'decoder_self_attn': self_att,
        # Static variable selection weights
        'static_flags': static_weights[Ellipsis, 0],
        # Variable selection weights of past inputs
        'historical_flags': historical_flags[Ellipsis, 0, :],
        # Variable selection weights of future inputs
        'future_flags': future_flags[Ellipsis, 0, :]
    }

    return transformer_layer, all_inputs, attention_components

  def build_model(self):
    """Build model and defines training losses.

    Returns:
      Fully defined Keras model.
    """

    with tf.compat.v1.variable_scope(self.name):

      transformer_layer, all_inputs, attention_components \
          = self._build_base_graph()

      outputs = tf.keras.layers.TimeDistributed(
          tf.keras.layers.Dense(self.output_size * len(self.quantiles))) \
          (transformer_layer[Ellipsis, self.num_encoder_steps:, :])

      self._attention_components = attention_components

      adam = tf.compat.v1.keras.optimizers.Adam(
          learning_rate=self.learning_rate, clipnorm=self.max_gradient_norm)

      model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

      print(model.summary())

      valid_quantiles = self.quantiles
      output_size = self.output_size

      class QuantileLossCalculator(object):
        """Computes the combined quantile loss for prespecified quantiles.

        Attributes:
          quantiles: Quantiles to compute losses
        """

        def __init__(self, quantiles):
          """Initializes computer with quantiles for loss calculations.

          Args:
            quantiles: Quantiles to use for computations.
          """
          self.quantiles = quantiles

        def quantile_loss(self, a, b):
          """Returns quantile loss for specified quantiles.

          Args:
            a: Targets
            b: Predictions
          """
          quantiles_used = set(self.quantiles)

          loss = 0.
          for i, quantile in enumerate(valid_quantiles):
            if quantile in quantiles_used:
              loss += utils.tensorflow_quantile_loss(
                  a[Ellipsis, output_size * i:output_size * (i + 1)],
                  b[Ellipsis, output_size * i:output_size * (i + 1)], quantile)
          return loss

      quantile_loss = QuantileLossCalculator(valid_quantiles).quantile_loss

      model.compile(
          loss=quantile_loss, optimizer=adam, sample_weight_mode='temporal')

      self._input_placeholder = all_inputs

    return model

  def fit(self, train_df=None, valid_df=None):
    """Fits deep neural network for given training and validation data.

    Args:
      train_df: DataFrame for training data
      valid_df: DataFrame for validation data
    """

    print('*** Fitting {} ***'.format(self.name))  # Muestra el nombre del modelo mientras comienza el entrenamiento

    # Agrega callbacks relevantes
    callbacks = [
        # EarlyStopping interrumpe el entrenamiento si el rendimiento no mejora después de ciertas épocas
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Observa la pérdida en los datos de validación
            patience=self.early_stopping_patience,  # Número de épocas sin mejora antes de parar
            min_delta=1e-4),  # Mínima diferencia para considerar que hay mejora

        # ModelCheckpoint guarda el modelo con mejor desempeño en el conjunto de validación
        tf.keras.callbacks.ModelCheckpoint(
            filepath=self.get_keras_saved_path(self._temp_folder),  # Ruta para guardar el modelo
            monitor='val_loss',  # Se guarda si la pérdida en validación mejora
            save_best_only=True,  # Solo se guarda el modelo con mejor rendimiento
            save_weights_only=True),  # Solo guarda los pesos, no toda la estructura del modelo

        # TerminateOnNaN termina el entrenamiento si hay valores NaN (lo que indica un problema numérico)
        tf.keras.callbacks.TerminateOnNaN()
    ]

    print('Getting batched_data')  # Muestra un mensaje cuando comienza a obtener los datos por lotes

    # Si no se pasan datos de entrenamiento, usa los datos en caché
    if train_df is None:
        print('Using cached training data')
        train_data = TFTDataCache.get('train')  # Obtiene los datos de entrenamiento almacenados
    else:
        train_data = self._batch_data(train_df)  # Si hay datos, los prepara por lotes

    # Si no se pasan datos de validación, usa los datos en caché
    if valid_df is None:
        print('Using cached validation data')
        valid_data = TFTDataCache.get('valid')  # Obtiene los datos de validación almacenados
    else:
        valid_data = self._batch_data(valid_df)  # Si hay datos, los prepara por lotes

    print('Using keras standard fit')  # Muestra un mensaje indicando que se usará la función estándar `fit` de Keras

    # Define una función para desempaquetar los datos, devolviendo entradas, salidas y banderas activas
    def _unpack(data):
        return data['inputs'], data['outputs'], self._get_active_locations(data['active_entries'])

    # Desempaquetar los datos de entrenamiento y validación sin pesos de muestra
    data, labels, active_flags = _unpack(train_data)  # Datos de entrada, etiquetas y banderas activas para entrenamiento
    val_data, val_labels, val_flags = _unpack(valid_data)  # Lo mismo para validación

    all_callbacks = callbacks  # Agrega los callbacks definidos previamente

    # Entrenamiento del modelo usando `fit` de Keras
    self.model.fit(
        x=data,  # Datos de entrada para entrenamiento
        y=np.concatenate([labels, labels, labels], axis=-1),  # Etiquetas concatenadas para las 3 predicciones (cuantiles)
        sample_weight=active_flags,  # Ponderación de los ejemplos de entrenamiento según banderas activas
        epochs=self.num_epochs,  # Número máximo de épocas
        batch_size=self.minibatch_size,  # Tamaño de minibatch para entrenamiento
        validation_data=(val_data, np.concatenate([val_labels, val_labels, val_labels], axis=-1), val_flags),  # Datos de validación
        callbacks=all_callbacks,  # Los callbacks para el monitoreo
        shuffle=True,  # Barajar los datos entre épocas
        use_multiprocessing=True,  # Habilitar multiproceso para acelerar el entrenamiento
        workers=self.n_multiprocessing_workers)  # Número de trabajadores para procesamiento paralelo

    # Cargar el mejor modelo guardado durante el entrenamiento (si existe)
    tmp_checkpont = self.get_keras_saved_path(self._temp_folder)
    if os.path.exists(tmp_checkpont):  # Verifica si el archivo de checkpoint existe
        self.load(self._temp_folder, use_keras_loadings=True)  # Carga el modelo guardado
    else:
        print('Cannot load from {}, skipping ...'.format(self._temp_folder))  # Si no se encuentra el archivo, se omite la carga
        
  def evaluate(self, data=None, eval_metric='loss'):
    """Applies evaluation metric to the training data.

    Args:
      data: Dataframe for evaluation
      eval_metric: Evaluation metric to return, based on model definition.

    Returns:
      Computed evaluation loss.
    """

    # Si no se proporcionan datos, usa los datos en caché (por defecto, datos de validación)
    if data is None:
        print('Using cached validation data')
        raw_data = TFTDataCache.get('valid')  # Obtiene los datos de validación desde la caché
    else:
        raw_data = self._batch_data(data)  # Si hay datos proporcionados, los procesa por lotes

    # Extrae las entradas, salidas y banderas activas desde los datos procesados
    inputs = raw_data['inputs']
    outputs = raw_data['outputs']
    active_entries = self._get_active_locations(raw_data['active_entries'])

    # Realiza la evaluación usando el modelo
    metric_values = self.model.evaluate(
        x=inputs,  # Datos de entrada
        y=np.concatenate([outputs, outputs, outputs], axis=-1),  # Las salidas (repetidas 3 veces para los 3 cuantiles)
        sample_weight=active_entries,  # Ponderación de las entradas activas
        workers=16,  # Número de trabajadores para el procesamiento paralelo
        use_multiprocessing=True)  # Habilitar el uso de multiproceso

    # Convierte los resultados en una serie de pandas con los nombres de las métricas
    metrics = pd.Series(metric_values, self.model.metrics_names)

    # Devuelve la métrica evaluada que se pidió (por defecto, la pérdida)
    return metrics[eval_metric]

  def predict(self, df, return_targets=False):
    """Computes predictions for a given input dataset.

    Args:
      df: Input dataframe
      return_targets: Whether to also return outputs aligned with predictions to
        facilitate evaluation

    Returns:
      Input dataframe or tuple of (input dataframe, aligned output dataframe).
    """

    # Procesa los datos de entrada por lotes
    data = self._batch_data(df)

    # Extrae las entradas, tiempos, identificadores y salidas de los datos
    inputs = data['inputs']
    time = data['time']
    identifier = data['identifier']
    outputs = data['outputs']

    # Genera las predicciones utilizando el modelo
    combined = self.model.predict(
        inputs,  # Datos de entrada
        workers=16,  # Número de trabajadores para procesamiento paralelo
        use_multiprocessing=True,  # Habilitar multiproceso
        batch_size=self.minibatch_size)  # Tamaño del lote para predicción

    # Si el tamaño de salida no es 1D, no está implementado
    if self.output_size != 1:
        raise NotImplementedError('Current version only supports 1D targets!')

    def format_outputs(prediction):
        """Returns formatted dataframes for prediction."""
        # Formatea las predicciones a un DataFrame de pandas
        flat_prediction = pd.DataFrame(
            prediction[:, :, 0],  # Extrae solo la primera dimensión de la predicción
            columns=[
                't+{}'.format(i)
                for i in range(self.time_steps - self.num_encoder_steps)  # Genera columnas para cada paso futuro
            ])
        # Añade las columnas de tiempo y el identificador
        flat_prediction['forecast_time'] = time[:, self.num_encoder_steps - 1, 0]
        flat_prediction['identifier'] = identifier[:, 0, 0]

        # Reorganiza las columnas de forma adecuada
        return flat_prediction[['forecast_time', 'identifier'] + list(flat_prediction.columns[:-2])]

    # Extrae las predicciones para cada cuantil en diferentes entradas
    process_map = {
        'p{}'.format(int(q * 100)):  # Genera una clave como 'p25', 'p50', etc., para cada cuantil
        combined[Ellipsis, i * self.output_size:(i + 1) * self.output_size]  # Predicción para cada cuantil
        for i, q in enumerate(self.quantiles)
    }

    if return_targets:
        # Si se requiere, agrega las salidas reales (targets) a las predicciones
        process_map['targets'] = outputs

    # Devuelve las predicciones procesadas, formateadas adecuadamente
    return {k: format_outputs(process_map[k]) for k in process_map}


  def get_attention(self, df):
    """Computes TFT attention weights for a given dataset.

    Args:
      df: Input dataframe

    Returns:
        Dictionary of numpy arrays for temporal attention weights and variable
          selection weights, along with their identifiers and time indices
    """

    data = self._batch_data(df)
    inputs = data['inputs']
    identifiers = data['identifier']
    time = data['time']

    def get_batch_attention_weights(input_batch):
      """Returns weights for a given minibatch of data."""
      input_placeholder = self._input_placeholder
      attention_weights = {}
      for k in self._attention_components:
        attention_weight = tf.compat.v1.keras.backend.get_session().run(
            self._attention_components[k],
            {input_placeholder: input_batch.astype(np.float32)})
        attention_weights[k] = attention_weight
      return attention_weights

    # Compute number of batches
    batch_size = self.minibatch_size
    n = inputs.shape[0]
    num_batches = n // batch_size
    if n - (num_batches * batch_size) > 0:
      num_batches += 1

    # Split up inputs into batches
    batched_inputs = [
        inputs[i * batch_size:(i + 1) * batch_size, Ellipsis]
        for i in range(num_batches)
    ]

    # Get attention weights, while avoiding large memory increases
    attention_by_batch = [
        get_batch_attention_weights(batch) for batch in batched_inputs
    ]
    attention_weights = {}
    for k in self._attention_components:
      attention_weights[k] = []
      for batch_weights in attention_by_batch:
        attention_weights[k].append(batch_weights[k])

      if len(attention_weights[k][0].shape) == 4:
        tmp = np.concatenate(attention_weights[k], axis=1)
      else:
        tmp = np.concatenate(attention_weights[k], axis=0)

      del attention_weights[k]
      gc.collect()
      attention_weights[k] = tmp

    attention_weights['identifiers'] = identifiers[:, 0, 0]
    attention_weights['time'] = time[:, :, 0]

    return attention_weights

  # Serialisation.
  def reset_temp_folder(self):
    """Deletes and recreates folder with temporary Keras training outputs."""
    print('Resetting temp folder...')
    utils.create_folder_if_not_exist(self._temp_folder)
    shutil.rmtree(self._temp_folder)
    os.makedirs(self._temp_folder)

  def get_keras_saved_path(self, model_folder):
    """Returns path to keras checkpoint."""
    return os.path.join(model_folder, '{}.check'.format(self.name))

  def save(self, model_folder):
    """Saves optimal TFT weights.

    Args:
      model_folder: Location to serialze model.
    """
    # Allows for direct serialisation of tensorflow variables to avoid spurious
    # issue with Keras that leads to different performance evaluation results
    # when model is reloaded (https://github.com/keras-team/keras/issues/4875).

    utils.save(
        tf.compat.v1.keras.backend.get_session(),
        model_folder,
        cp_name=self.name,
        scope=self.name)

  def load(self, model_folder, use_keras_loadings=False):
    """Loads TFT weights.

    Args:
      model_folder: Folder containing serialized models.
      use_keras_loadings: Whether to load from Keras checkpoint.

    Returns:

    """
    if use_keras_loadings:
      # Loads temporary Keras model saved during training.
      serialisation_path = self.get_keras_saved_path(model_folder)
      print('Loading model from {}'.format(serialisation_path))
      self.model.load_weights(serialisation_path)
    else:
      # Loads tensorflow graph for optimal models.
      utils.load(
          tf.compat.v1.keras.backend.get_session(),
          model_folder,
          cp_name=self.name,
          scope=self.name)

  @classmethod
  def get_hyperparm_choices(cls):
    """Returns hyperparameter ranges for random search."""
    return {
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
        'hidden_layer_size': [10, 20, 40, 80, 160, 240, 320],
        'minibatch_size': [64, 128, 256],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'max_gradient_norm': [0.01, 1.0, 100.0],
        'num_heads': [1, 4],
        'stack_size': [1],
    }
