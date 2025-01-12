import os
import re
import numpy as np
import tensorflow as tf


class CustomEmbedding(tf.keras.layers.Layer):
    """
    A custom embedding layer that learns an embedding for each feature and applies a
    dense layer with a ReLU activation for further transformation.

    This layer performs an embedding lookup for each feature based on a learned set of
    embedding weights. The input is expected to have a shape of [batch_size, nfeatures],
    and the output will have the shape [batch_size, output_dim], where `output_dim` is
    the dimensionality of the embedding.

    Parameters:
    -----------
    input_dim : int
        The size of the input feature space (i.e., the number of possible input features).
    output_dim : int
        The dimensionality of the embedding space (i.e., the size of the output representation).

    Methods:
    --------
    build(input_shape):
        Initializes the embedding weights and the dense layer.
    call(inputs):
        Performs the embedding lookup and applies a dense layer with ReLU activation.
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        """
        Initializes the CustomEmbedding layer with input and output dimensions.

        Parameters:
        -----------
        input_dim : int
            The size of the input feature space (number of possible input features).
        output_dim : int
            The size of the embedding output space (embedding dimension).
        """
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        """
        Initializes the layer's trainable weights and creates the dense layer.

        This method is called during model building, and it initializes:
        - The embedding weights with shape (input_dim, output_dim) using a random normal initializer.
        - A dense layer with ReLU activation for transforming the embedding outputs.

        Parameters:
        -----------
        input_shape : tuple
            The shape of the input tensor (not used directly here, as it's inferred from the input dimensions).
        """
        self.embedding_weights = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer="random_normal",
            trainable=True,
        )
        self.dense_layer = tf.keras.layers.Dense(self.output_dim, activation="relu")

    def call(self, inputs):
        """
        Perform embedding lookup and apply a dense layer to the inputs.

        Parameters:
        -----------
        inputs : tf.Tensor
            The input tensor with shape [batch_size, nfeatures], where each feature corresponds
            to an index that will be mapped to an embedding.

        Returns:
        --------
        tf.Tensor
            The embedded representation of the input with shape [batch_size, output_dim]
            after applying the dense layer.

        Raises:
        -------
        ValueError:
            If the last dimension of the input does not match `input_dim`, an error is raised.
        """
        # Validate input shape
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected inputs with last dimension {self.input_dim}, "
                f"but got {inputs.shape[-1]}"
            )

        # Perform the embedding lookup via matrix multiplication
        # inputs: [batch_size, nfeatures]
        embedded = tf.matmul(inputs, self.embedding_weights)  # [batch_size, output_dim]

        # Apply a Dense layer with non-linearity for more flexibility
        embedded = self.dense_layer(embedded)

        return embedded


class SmartEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, max_cardinality=20, use_embedding=True, **kwargs):
        """
        Automatically handles mixed input types (categorical and continuous).

        Parameters:
        -----------
        embedding_dim : int
            Dimensionality of embedding output for categorical features.
        max_cardinality : int
            Maximum number of unique values to consider a feature as categorical.
        use_embedding : bool
            Whether to use embeddings for categorical features. If False, one-hot encoding is applied.
        """
        super(SmartEmbedding, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_cardinality = max_cardinality
        self.use_embedding = use_embedding
        self.feature_types = []  # Placeholder to store detected types
        self.continuous_dense = None
        self.categorical_layers = {}

    def build(self, input_shape):
        """
        Initializes the transformation layers after detecting feature types.
        """
        nfeatures = input_shape[-1]

        # Layers to transform continuous features
        self.continuous_dense = tf.keras.layers.Dense(
            self.embedding_dim, activation=None
        )

        # Placeholder lists for feature handling
        self.continuous_indices = []
        self.categorical_indices = []
        self.categorical_cardinalities = []

    def detect_feature_types(self, inputs):
        """
        Detects the types of each feature (categorical or continuous).
        """
        for i in range(inputs.shape[-1]):
            feature_column = inputs[:, i]
            unique_values = tf.unique(feature_column)[0]
            unique_count = tf.size(unique_values)

            # Detect categorical or continuous
            if unique_count <= self.max_cardinality:  # Likely categorical
                self.feature_types.append("categorical")
                self.categorical_indices.append(i)
                self.categorical_cardinalities.append(unique_count)
            else:
                self.feature_types.append("continuous")
                self.continuous_indices.append(i)

        # Create embedding layers for categorical features
        if self.use_embedding:
            self.categorical_layers = [
                tf.keras.layers.Embedding(
                    input_dim=cardinality,
                    output_dim=min(self.embedding_dim, cardinality // 2 + 1),
                )
                for cardinality in self.categorical_cardinalities
            ]

    def call(self, inputs, training=None):
        """
        Applies transformations to both categorical and continuous features.

        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor of shape [batch_size, nfeatures].
        training : bool
            Training mode flag.

        Returns:
        --------
        tf.Tensor
            Concatenated representation of continuous and categorical features.
        """
        # Detect feature types if not already done
        if not self.feature_types:
            self.detect_feature_types(inputs)

        # Extract and transform continuous features
        if self.continuous_indices:
            continuous_inputs = tf.gather(inputs, self.continuous_indices, axis=-1)
            continuous_embeddings = self.continuous_dense(continuous_inputs)
        else:
            continuous_embeddings = None

        # Extract and transform categorical features
        if self.categorical_indices:
            categorical_inputs = tf.gather(inputs, self.categorical_indices, axis=-1)
            if self.use_embedding:
                categorical_embeddings = [
                    self.categorical_layers[i](
                        tf.cast(categorical_inputs[:, i], tf.int32)
                    )
                    for i in range(len(self.categorical_indices))
                ]
            else:
                categorical_embeddings = [
                    tf.one_hot(
                        tf.cast(categorical_inputs[:, i], tf.int32),
                        depth=self.categorical_cardinalities[i],
                    )
                    for i in range(len(self.categorical_indices))
                ]
            categorical_embeddings = tf.concat(categorical_embeddings, axis=-1)
        else:
            categorical_embeddings = None

        # Combine continuous and categorical embeddings
        if continuous_embeddings is not None and categorical_embeddings is not None:
            return tf.concat([continuous_embeddings, categorical_embeddings], axis=-1)
        elif continuous_embeddings is not None:
            return continuous_embeddings
        elif categorical_embeddings is not None:
            return categorical_embeddings
        else:
            raise ValueError("No features detected to process.")


def vect_embedding(
    input_tensor, vect_max_tokens, vect_output_length, voc, embedding_dim
):
    """
    Vectorizes and embeds text input using TensorFlow's `TextVectorization` and `Embedding` layers.

    This function performs the following steps:
    1. Vectorizes the text input using the `TextVectorization` layer, which converts raw text into integer sequences.
    2. Adapts the vectorizer to the provided vocabulary (`voc`).
    3. Embeds the vectorized sequences using the `Embedding` layer, mapping integers to dense vectors of fixed size.

    Parameters:
    ----------
    input_tensor : tf.Tensor
        A tensor containing raw text input to be processed.
    vect_max_tokens : int
        The maximum size of the vocabulary for the `TextVectorization` layer.
    vect_output_length : int
        The fixed length of the output sequences from the `TextVectorization` layer.
        Shorter sequences will be padded, and longer sequences will be truncated.
    voc : list
        A list of vocabulary words to adapt the `TextVectorization` layer. This is used to map words to indices.
    embedding_dim : int
        The dimensionality of the dense embedding vectors.

    Returns:
    -------
    tf.Tensor
        A tensor containing the embedded representation of the input text. The shape of the tensor is
        `(batch_size, vect_output_length, embedding_dim)`.
    """

    # Text vectorization
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vect_max_tokens,
        output_mode="int",
        output_sequence_length=vect_output_length,
    )

    # Adapt the vectorizers to the actual training data (raw text data)
    vectorizer.adapt(voc)
    input_vectorized = vectorizer(input_tensor)

    # Embedding
    embedding = tf.keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()), output_dim=embedding_dim
    )
    input_embedded = embedding(input_vectorized)

    return input_embedded


def regulation_block(input_layer, vdropout, spatial_dropout=False):
    """
    Applies batch normalization and dropout to the input layer.

    This function is used to regularize a neural network by first normalizing the input
    features using batch normalization and then applying dropout to prevent overfitting.

    Parameters:
    ----------
    input_layer : tf.Tensor
        The input tensor or layer to which the regularization operations will be applied.
    vdropout : float
        The dropout rate, a value between 0 and 1, representing the fraction of input units to drop.
    spatial_dropout : bool, optional (default=False)
        Whether to apply spatial dropout (recommended for embedding layers).

    Returns:
    -------
    tf.Tensor
        A tensor resulting from the application of batch normalization followed by dropout on the input tensor.
    """
    # Apply Batch Normalization
    h = tf.keras.layers.BatchNormalization()(input_layer)

    # Apply Spatial Dropout for embeddings (recommended for text embeddings)
    if spatial_dropout:
        h = tf.keras.layers.SpatialDropout1D(vdropout)(
            h
        )  # Use for sequence-based inputs (e.g., text)

    # Apply regular Dropout
    else:
        h = tf.keras.layers.Dropout(vdropout)(h)

    return h


def build_DNN(
    nfeatures: int,
    nDlayers: int,
    vdropout: float,
    nodes_per_layer: int,
    act_fn: str,
    nclass: int,
    embedding: bool,
    embedding_dim: int,
) -> tf.keras.Model:

    # Instantiate a Keras input tensors
    input_tensor = tf.keras.Input(shape=(nfeatures,))

    # Include embedding - To be completed
    if embedding == True:

        # Create embedding layer
        embedding_layer = CustomEmbedding(input_dim=nfeatures, output_dim=embedding_dim)

        # Apply embedding layer
        x = embedding_layer(input_tensor)
        # x = regulation_block(x, vdropout, spatial_dropout=False)

    else:
        x = input_tensor

    h = tf.keras.layers.Dense(nodes_per_layer, activation=act_fn)(x)
    h = regulation_block(h, vdropout, spatial_dropout=False)

    # Add n-1 hidden layers
    for _ in range(nDlayers - 1):
        nodes_per_layer = nodes_per_layer
        h = regulation_block(h, vdropout)
        h = tf.keras.layers.Dense(
            nodes_per_layer,
            activation=act_fn,
        )(h)

    # Add dropout to the final dense layer
    h = regulation_block(h, vdropout, spatial_dropout=False)

    # Add output layer for classification problems
    out = tf.keras.layers.Dense(nclass, activation="softmax")(h)

    # Create and retrun the model
    model = tf.keras.Model(inputs=input_tensor, outputs=out)

    return model


def build_NLP(
    numerical_features_dim: int,
    embedding_dim: int,
    mha: bool,
    attention_heads: int,
    nDlayers: int,
    vdropout: float,
    nodes_per_layer: int,
    act_fn: str,
    nclass: int,
    title_voc: str,
    body_voc: str,
    tags_voc: str,
) -> tf.keras.Model:
    """
    Builds a neural network model for natural language processing (NLP) tasks.

    This function creates an NLP model that incorporates both textual and numerical data,
    using text embeddings, optional multi-head attention, and dense layers for classification.

    Parameters:
    ----------
    numerical_features_dim : int
        The dimensionality of the numerical features.
    embedding_dim : int
        The dimension of the word embeddings used for the textual inputs.
    mha : bool
        Whether to apply Multi-Head Attention to combine the text features (True) or use concatenation (False).
    attention_heads : int
        The number of attention heads used in the Multi-Head Attention layer (used only if `mha=True`).
    nDlayers : int
        The number of dense layers to include in the fully connected part of the model.
    vdropout : float
        The dropout rate to be applied after regularization layers.
    nodes_per_layer : int
        The number of nodes in each dense layer of the fully connected part of the model.
    act_fn : str
        The activation function to use in the dense layers (e.g., 'relu', 'tanh').
    nclass : int
        The number of output classes for the classification task.
    title_voc : str
        The vocabulary file for the title input text.
    body_voc : str
        The vocabulary file for the body input text.
    tags_voc : str
        The vocabulary file for the tags input text.

    Returns:
    -------
    tf.keras.Model
        A compiled Keras model ready for training or inference, designed to handle both textual and numerical data inputs.
    """

    # Inputs
    title_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="title_input")
    body_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="body_input")
    tags_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="tags_input")
    numerical_input = tf.keras.Input(
        shape=(numerical_features_dim,), name="numerical_input"
    )

    title_embedded = vect_embedding(
        input_tensor=title_input,
        vect_max_tokens=10000,
        vect_output_length=20,
        voc=title_voc,
        embedding_dim=embedding_dim,
    )
    body_embedded = vect_embedding(
        input_tensor=body_input,
        vect_max_tokens=10000,
        vect_output_length=20,
        voc=body_voc,
        embedding_dim=embedding_dim,
    )
    tags_embedded = vect_embedding(
        input_tensor=tags_input,
        vect_max_tokens=10000,
        vect_output_length=20,
        voc=tags_voc,
        embedding_dim=embedding_dim,
    )

    title_embedded = regulation_block(title_embedded, vdropout, spatial_dropout=True)
    body_embedded = regulation_block(body_embedded, vdropout, spatial_dropout=True)
    tags_embedded = regulation_block(tags_embedded, vdropout, spatial_dropout=True)

    if mha == True:
        # Multi-head Attention for Text
        text_attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=attention_heads, key_dim=embedding_dim
        )
        text_combined = text_attention_layer(
            query=body_embedded, key=tags_embedded, value=title_embedded
        )
        text_combined = regulation_block(text_combined, vdropout, spatial_dropout=False)

    else:
        text_combined = tf.keras.layers.Concatenate()(
            [title_embedded, body_embedded, tags_embedded]
        )
        text_combined = regulation_block(text_combined, vdropout, spatial_dropout=False)

    # Flatten and Concatenate Features
    text_flattened = tf.keras.layers.GlobalAveragePooling1D()(text_combined)
    combined_features = tf.keras.layers.Concatenate()([text_flattened, numerical_input])
    # combined_features = text_flattened

    # Fully Connected Layers
    h = tf.keras.layers.Dense(nodes_per_layer, activation=act_fn)(combined_features)
    h = regulation_block(h, vdropout, spatial_dropout=False)

    # Add n-1 hidden layers
    for _ in range(nDlayers - 1):
        h = tf.keras.layers.Dense(nodes_per_layer, activation=act_fn)(h)
        h = regulation_block(h, vdropout, spatial_dropout=False)

    # Add output layer for classification
    output_layer = tf.keras.layers.Dense(
        nclass, activation="softmax", name="output_layer"
    )(h)

    # Create and return the model
    model = tf.keras.Model(
        inputs=[title_input, body_input, tags_input, numerical_input],
        outputs=output_layer,
    )

    return model


def get_latest_checkpoint(checkpoint_dir):
    """
    Get the latest model checkpoint file.

    Parameters:
    - checkpoint_dir (str): Directory containing model checkpoints.

    Returns:
    - str or None: The path to the latest checkpoint file, or None if no checkpoints are found.
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")]
    if not checkpoints:
        return None
    latest_checkpoint = max(
        checkpoints, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )
    return os.path.join(checkpoint_dir, latest_checkpoint)


def get_best_checkpoint(dir_path):
    """
    Get the best model checkpoint based on validation accuracy.

    Parameters:
    - dir_path (str): The path to the directory containing checkpoint files.

    Returns:
    - str: The file path of the best checkpoint, or an empty string if none is found.
    """
    best_checkpoint = None
    best_val_acc = 0.0

    for filename in os.listdir(dir_path):
        # Check if the file is a checkpoint file
        if not filename.endswith(".keras"):
            continue

        # Extract the validation accuracy from the file name
        match = re.search(r"acc-(\d+\.\d+)-(\d+\.\d+)", filename)
        if not match:
            continue

        val_acc = float(match.group(2))
        if val_acc > best_val_acc:
            best_checkpoint = os.path.join(dir_path, filename)
            best_val_acc = val_acc

    print("Selected model checkpoint: ", best_checkpoint)

    return best_checkpoint if best_checkpoint else ""
