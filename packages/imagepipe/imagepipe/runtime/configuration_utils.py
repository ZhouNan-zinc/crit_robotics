import copy
import json
import math
import os
from typing import TYPE_CHECKING, Any, TypeVar, Union

from .utils import logging
from .utils import (
    CONFIG_NAME,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# type hinting: specifying the type of config class that inherits from PreTrainedConfig
SpecificPreTrainedConfigType = TypeVar("SpecificPreTrainedConfigType", bound="PreTrainedConfig")

_FLOAT_TAG_KEY = "__float__"
_FLOAT_TAG_VALUES = {"Infinity": float("inf"), "-Infinity": float("-inf"), "NaN": float("nan")}


class PreTrainedConfig:
    # no-format
    r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    <Tip>

    A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
    initialize a model does **not** load the model weights. It only affects the model's configuration.

    </Tip>

    Class attributes (overridden by derived classes):

    - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, and used to recreate
      the correct object in [`~transformers.AutoConfig`].
    - **has_no_defaults_at_init** (`bool`) -- Whether the config class can be initialized without providing input arguments.
      Some configurations requires inputs to be defined at init and have no default values, usually these are composite configs,
      (but not necessarily) such as [`~transformers.EncoderDecoderConfig`] or [`~RagConfig`]. They have to be initialized from
      two or more configs of type [`~transformers.PreTrainedConfig`].
    - **keys_to_ignore_at_inference** (`list[str]`) -- A list of keys to ignore by default when looking at dictionary
      outputs of the model during inference.
    - **attribute_map** (`dict[str, str]`) -- A dict that maps model specific attribute names to the standardized
      naming of attributes.
    - **base_model_tp_plan** (`dict[str, Any]`) -- A dict that maps sub-modules FQNs of a base model to a tensor
      parallel plan applied to the sub-module when `model.tensor_parallel` is called.
    - **base_model_pp_plan** (`dict[str, tuple[list[str]]]`) -- A dict that maps child-modules of a base model to a
      pipeline parallel plan that enables users to place the child-module on the appropriate device.

    Common attributes (present in all subclasses):

    - **vocab_size** (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
      embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
    - **hidden_size** (`int`) -- The hidden size of the model.
    - **num_attention_heads** (`int`) -- The number of attention heads used in the multi-head attention layers of the
      model.
    - **num_hidden_layers** (`int`) -- The number of blocks in the model.

    <Tip warning={true}>

    Setting parameters for sequence generation in the model config is deprecated. For backward compatibility, loading
    some of them will still be possible, but attempting to overwrite them will throw an exception -- you should set
    them in a [~transformers.GenerationConfig]. Check the documentation of [~transformers.GenerationConfig] for more
    information about the individual parameters.

    </Tip>

    Arg:
        name_or_path (`str`, *optional*, defaults to `""`):
            Store the string that was passed to [`PreTrainedModel.from_pretrained`] as `pretrained_model_name_or_path`
            if the configuration was created with such a method.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return all hidden-states.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not the model should returns all attentions.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a [`~transformers.utils.ModelOutput`] instead of a plain tuple.
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether to only use the decoder in an encoder-decoder architecture, otherwise it has no effect on
            decoder-only or encoder-only architectures.
        cross_attention_hidden_size (`bool`, *optional*):
            The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder
            setting and the cross-attention hidden dimension differs from `self.config.hidden_size`.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the [`EncoderDecoderModel`] class, which consists of all models
            in `AUTO_MODELS_FOR_CAUSAL_LM`.
        chunk_size_feed_forward (`int`, *optional*, defaults to `0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that
            the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` <
            sequence_length embeddings at a time. For more information on feed forward chunking, see [How does Feed
            Forward Chunking work?](../glossary.html#feed-forward-chunking).

        > Parameters for fine-tuning tasks

        architectures (`list[str]`, *optional*):
            Model architectures that can be used with the model pretrained weights.
        finetuning_task (`str`, *optional*):
            Name of the task used to fine-tune the model.
        id2label (`dict[int, str]`, *optional*):
            A map from index (for instance prediction index, or target index) to label.
        label2id (`dict[str, int]`, *optional*):
            A map from label to index for the model.
        num_labels (`int`, *optional*):
            Number of labels to use in the last layer added to the model, typically for a classification task.
        task_specific_params (`dict[str, Any]`, *optional*):
            Additional keyword arguments to store for the current task.
        problem_type (`str`, *optional*):
            Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.

        > Parameters linked to the tokenizer

        tokenizer_class (`str`, *optional*):
            The name of the associated tokenizer class to use (if none is set, will use the tokenizer associated to the
            model by default).
        prefix (`str`, *optional*):
            A specific prompt that should be added at the beginning of each text before calling the model.
        bos_token_id (`int`, *optional*):
            The id of the _beginning-of-stream_ token.
        pad_token_id (`int`, *optional*):
            The id of the _padding_ token.
        eos_token_id (`int`, *optional*):
            The id of the _end-of-stream_ token.
        decoder_start_token_id (`int`, *optional*):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token.
        sep_token_id (`int`, *optional*):
            The id of the _separation_ token.

        > PyTorch specific parameters

        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        dtype (`str`, *optional*):
            The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype`
            (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved
            model is `float16`, ideally we want to load it back using the minimal amount of memory needed to load
            `float16` weights.
    """

    base_config_key: str = ""
    sub_configs: dict[str, type["PreTrainedConfig"]] = {}
    attribute_map: dict[str, str] = {}

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(
        self,
        *,
        # All models common arguments
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        dtype: Union[str, "torch.dtype"] | None = None,
        # Common arguments
        tie_word_embeddings: bool = True,
        chunk_size_feed_forward: int = 0,
        is_encoder_decoder: bool = False,
        is_decoder: bool = False,
        cross_attention_hidden_size: int | None = None,
        add_cross_attention: bool = False,
        # Fine-tuning task arguments
        architectures: list[str] | None = None,
        finetuning_task: str | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        num_labels: int | None = None,
        task_specific_params: dict[str, Any] | None = None,
        problem_type: str | None = None,
        # Tokenizer kwargs
        tokenizer_class: str | None = None,
        prefix: str | None = None,
        bos_token_id: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        sep_token_id: int | None = None,
        decoder_start_token_id: int | None = None,
        **kwargs,
    ):
        # Validation for some arguments
        if label2id is not None and not isinstance(label2id, dict):
            raise ValueError("Argument label2id should be a dictionary.")
        if id2label is not None and not isinstance(id2label, dict):
            raise ValueError("Argument id2label should be a dictionary.")
        if num_labels is not None and id2label is not None and len(id2label) != num_labels:
            logger.warning(
                f"You passed `num_labels={num_labels}` which is incompatible to "
                f"the `id2label` map of length `{len(id2label)}`."
            )
        if problem_type is not None and problem_type not in (
            "regression",
            "single_label_classification",
            "multi_label_classification",
        ):
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )
        # BC for the `torch_dtype` argument instead of the simpler `dtype`
        # Do not warn, as it would otherwise always be triggered since most configs on the hub have `torch_dtype`
        if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
            # If both are provided, keep `dtype`
            dtype = dtype if dtype is not None else torch_dtype
        if dtype is not None and isinstance(dtype, str):
            # we will start using self.dtype in v5, but to be consistent with
            # from_pretrained's dtype arg convert it to an actual torch.dtype object
            import torch

            dtype = getattr(torch, dtype)

        # BC for rotary embeddings. We will pop out legacy keys from kwargs and rename to new format
        if hasattr(self, "rope_parameters"):
            ignore_keys_at_rope_validation = kwargs.pop("ignore_keys_at_rope_validation", None)
            kwargs = self.convert_rope_params_to_dict(
                ignore_keys_at_rope_validation=ignore_keys_at_rope_validation, **kwargs
            )

        # Attributes common for all models
        self.return_dict = return_dict
        self.output_hidden_states = output_hidden_states
        self.dtype = dtype
        self._output_attentions = output_attentions  # has public property

        # Less common kwargs, only used by some models
        if "tie_encoder_decoder" in kwargs:
            tie_encoder_decoder = kwargs.pop("tie_encoder_decoder")
            tie_word_embeddings = tie_encoder_decoder or tie_word_embeddings

        self.tie_word_embeddings = tie_word_embeddings
        self.chunk_size_feed_forward = chunk_size_feed_forward

        # Encoder-decoder models attributes
        self.is_encoder_decoder = is_encoder_decoder
        self.is_decoder = is_decoder  # used in encoder-decoder models to differentiate encoder from decoder
        self.cross_attention_hidden_size = cross_attention_hidden_size
        self.add_cross_attention = add_cross_attention

        # Fine-tuning task attributes
        self.architectures = architectures
        self.finetuning_task = finetuning_task
        self.id2label = id2label
        self.label2id = label2id
        self.task_specific_params = task_specific_params
        self.problem_type = problem_type

        if self.id2label is None:
            self._create_id_label_maps(num_labels if num_labels is not None else 2)
        else:
            # Keys are always strings in JSON so convert ids to int here.
            self.id2label = {int(key): value for key, value in self.id2label.items()}

        # Tokenizer attributes
        self.tokenizer_class = tokenizer_class
        self.prefix = prefix
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.decoder_start_token_id = decoder_start_token_id

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        self._commit_hash = kwargs.pop("_commit_hash", None)

        # Attention implementation to use, if relevant (it sets it recursively on sub-configs)
        self._attn_implementation = kwargs.pop("attn_implementation", None)

        # Experts implementation to use, if relevant (it sets it recursively on sub-configs)
        self._experts_implementation = kwargs.pop("experts_implementation", None)

        # Drop the transformers version info
        self.transformers_version = kwargs.pop("transformers_version", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _create_id_label_maps(self, num_labels: int):
        self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    @property
    def name_or_path(self) -> str | None:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def output_attentions(self):
        """
        `bool`: Whether or not the model should returns all attentions.
        """
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(self, value: bool):
        # If we set `output_attentions` explicitly before the attn implementation, dispatch eager
        if value and self._attn_implementation is None:
            self._attn_implementation = "eager"
        if value and self._attn_implementation != "eager":
            raise ValueError(
                "The `output_attentions` attribute is not supported when using the `attn_implementation` set to "
                f"{self._attn_implementation}. Please set it to 'eager' instead."
            )
        self._output_attentions = value

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
        """
        return self.return_dict

    @property
    def num_labels(self) -> int:
        """
        `int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        # we do not store `num_labels` attribute in config, but instead
        # compute it based on the length of the `id2label` map
        if self.id2label is None or self.num_labels != num_labels:
            self._create_id_label_maps(num_labels)

    @classmethod
    def from_pretrained(
        cls: type[SpecificPreTrainedConfigType],
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> SpecificPreTrainedConfigType:
        r"""
        Instantiate a [`PreTrainedConfig`] (or a derived class) from a pretrained model configuration.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~PreTrainedConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`PreTrainedConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        # We can't instantiate directly the base class *PreTrainedConfig* so let's show the examples on a
        # derived class: BertConfig
        config = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased"
        )  # Download configuration from huggingface.co and cache.
        config = BertConfig.from_pretrained(
            "./test/saved_model/"
        )  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
        config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
        config = BertConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        assert config.output_attentions == True
        config, unused_kwargs = BertConfig.from_pretrained(
            "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        )
        assert config.output_attentions == True
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            # sometimes the config has no `base_config_key` if the config is used in several composite models
            # e.g. LlamaConfig. In that case we try to see if there is match in `model_type` before raising a warning
            for v in config_dict.values():
                if isinstance(v, dict) and v.get("model_type") == cls.model_type:
                    config_dict = v

            # raise warning only if we still can't see a match in `model_type`
            if config_dict["model_type"] != cls.model_type:
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )

        return cls.from_dict(config_dict, **kwargs)


    def get_text_config(self, decoder=None, encoder=None) -> "PreTrainedConfig":
        """
        Returns the text config related to the text input (encoder) or text output (decoder) of the model. The
        `decoder` and `encoder` input arguments can be used to specify which end of the model we are interested in,
        which is useful on models that have both text input and output modalities.

        There are three possible outcomes of using this method:
        1. On most models, it returns the original config instance itself.
        2. On newer (2024+) composite models, it returns the text section of the config, which is nested under a set
            of valid names.
        3. On older (2023-) composite models, it discards decoder-only parameters when `encoder=True` and vice-versa.

        Args:
            decoder (`Optional[bool]`, *optional*):
                If set to `True`, then only search for decoder config names.
            encoder (`Optional[bool]`, *optional*):
                If set to `True`, then only search for encoder config names.
        """
        return_both = decoder == encoder  # both unset or both set -> search all possible names

        decoder_possible_text_config_names = ("decoder", "generator", "text_config")
        encoder_possible_text_config_names = ("text_encoder",)
        if return_both:
            possible_text_config_names = encoder_possible_text_config_names + decoder_possible_text_config_names
        elif decoder:
            possible_text_config_names = decoder_possible_text_config_names
        else:
            possible_text_config_names = encoder_possible_text_config_names

        valid_text_config_names = []
        for text_config_name in possible_text_config_names:
            if hasattr(self, text_config_name):
                text_config = getattr(self, text_config_name, None)
                if text_config is not None:
                    valid_text_config_names += [text_config_name]

        if len(valid_text_config_names) > 1:
            raise ValueError(
                f"Multiple valid text configs were found in the model config: {valid_text_config_names}. In this "
                "case, using `get_text_config()` would be ambiguous. Please specify the desired text config directly, "
                "e.g. `text_config = config.sub_config_name`"
            )
        elif len(valid_text_config_names) == 1:
            config_to_return = getattr(self, valid_text_config_names[0])
        else:
            config_to_return = self

        # handle legacy models with flat config structure, when we only want one of the configs
        if not return_both and len(valid_text_config_names) == 0 and config_to_return.is_encoder_decoder:
            config_to_return = copy.deepcopy(config_to_return)
            prefix_to_discard = "encoder" if decoder else "decoder"
            prefix_to_keep = "decoder" if decoder else "encoder"
            for key in config_to_return.to_dict():
                # NOTE: We don't want to discard the key if it is mapped from a different attribute name at read time
                if key.startswith(prefix_to_discard) and key not in config_to_return.attribute_map.values():
                    delattr(config_to_return, key)
                if key.startswith(prefix_to_keep):
                    # [encoder/decoder]_layers -> num_hidden_layers
                    if key == prefix_to_keep + "_layers":
                        new_key = "num_hidden_layers"
                    # [encoder/decoder]_attention_heads -> num_attention_heads
                    elif key == prefix_to_keep + "_attention_heads":
                        new_key = "num_attention_heads"
                    # e.g. encoder_hidden_act -> hidden_act
                    else:
                        new_key = key[len(prefix_to_keep) + 1 :]

                    # Does the class map the new key into a different attribute name at read time? if so, let's write
                    # into that attribute instead
                    if new_key in config_to_return.attribute_map:
                        new_key = config_to_return.attribute_map[new_key]

                    value = getattr(config_to_return, key)
                    delattr(config_to_return, key)
                    setattr(config_to_return, new_key, value)

        return config_to_return


    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PreTrainedConfig`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict is None:
            return {}, kwargs
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            configuration_file = config_dict["configuration_files"]
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        
        if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            # Special case when pretrained_model_name_or_path is a local file
            resolved_config_file = pretrained_model_name_or_path
        else:
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME)
            try:
                # Load from local folder or from cache or download from model Hub and cache
                
                if os.path.exists(os.path.join(pretrained_model_name_or_path, configuration_file)):
                    resolved_config_file = os.path.join(pretrained_model_name_or_path, configuration_file)
                elif os.path.exists(os.path.join(pretrained_model_name_or_path, configuration_file)):
                    resolved_config_file = os.path.join(pretrained_model_name_or_path, configuration_file)
                else:
                    return None, kwargs
            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. "
                    f"Make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {configuration_file} file. "
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OSError(f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file.")

        logger.info(f"loading configuration file {resolved_config_file}")

        return config_dict, kwargs

    @classmethod
    def from_dict(
        cls: type[SpecificPreTrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> SpecificPreTrainedConfigType:
        """
        Instantiates a [`PreTrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PreTrainedConfig.get_config_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PreTrainedConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # For BC on the old `torch_dtype`
        if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
            logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
            # If both are present, use `dtype`
            kwargs["dtype"] = kwargs.get("dtype", torch_dtype)

        # We remove them from kwargs so that they do not appear in `return_unused_kwargs`.
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)
        config_dict["experts_implementation"] = kwargs.pop("experts_implementation", None)

        config = cls(**config_dict)

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # To authorize passing a custom subconfig as kwarg in models that have nested configs.
                # We need to update only custom kwarg values instead and keep other attributes in subconfig.
                if isinstance(current_attr, PreTrainedConfig) and isinstance(value, dict):
                    current_attr_updated = current_attr.to_dict()
                    current_attr_updated.update(value)
                    value = current_attr.__class__(**current_attr_updated)
                setattr(config, key, value)
                if key != "dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(
        cls: type[SpecificPreTrainedConfigType], json_file: str | os.PathLike
    ) -> SpecificPreTrainedConfigType:
        """
        Instantiates a [`PreTrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PreTrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)

        return cls._decode_special_floats(config_dict)

    @classmethod
    def _encode_special_floats(cls, obj: Any) -> Any:
        """
        Iterates over the passed object and encode specific floats that cannot be JSON-serialized. Python's JSON
        engine saves floats like `Infinity` (+/-) or `NaN` which are not compatible with other JSON engines.

        It serializes floats like `Infinity` as an object: `{'__float__': Infinity}`.
        """
        if isinstance(obj, float):
            if math.isnan(obj):
                return {_FLOAT_TAG_KEY: "NaN"}
            if obj == float("inf"):
                return {_FLOAT_TAG_KEY: "Infinity"}
            if obj == float("-inf"):
                return {_FLOAT_TAG_KEY: "-Infinity"}
            return obj

        if isinstance(obj, dict):
            return {k: cls._encode_special_floats(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [cls._encode_special_floats(v) for v in obj]

        return obj

    @classmethod
    def _decode_special_floats(cls, obj: Any) -> Any:
        """
        Iterates over the passed object and decode specific floats that cannot be JSON-serialized. Python's JSON
        engine saves floats like `Infinity` (+/-) or `NaN` which are not compatible with other JSON engines.

        This method deserializes objects like `{'__float__': Infinity}` to their float values like `Infinity`.
        """
        if isinstance(obj, dict):
            if set(obj.keys()) == {_FLOAT_TAG_KEY} and isinstance(obj[_FLOAT_TAG_KEY], str):
                tag = obj[_FLOAT_TAG_KEY]
                if tag in _FLOAT_TAG_VALUES:
                    return _FLOAT_TAG_VALUES[tag]
                return obj

            return {k: cls._decode_special_floats(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [cls._decode_special_floats(v) for v in obj]

        return obj

    def __eq__(self, other):
        return isinstance(other, PreTrainedConfig) and (self.__dict__ == other.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def __iter__(self):
        yield from self.__dict__

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from the configuration that correspond to the default config attributes for
        better readability, while always retaining the `config` attribute from the class. Serializes to a
        Python dictionary.

        Returns:
            dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        config_dict = self.to_dict()

        # Get the default config dict (from a fresh PreTrainedConfig instance)
        default_config_dict = PreTrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict()

        serializable_config_dict = {}

        # Only serialize values that differ from the default config,
        # except always keep the 'config' attribute.
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PreTrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # For nested configs we need to clean the diff recursively
                diff = recursive_diff_dict(value, default_config_dict, config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]

                serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or key == "vocab_file"
                or value != default_config_dict[key]
                or (key in default_config_dict and value != class_config_dict.get(key, value))
            ):
                serializable_config_dict[key] = value

        self._remove_keys_not_serialized(serializable_config_dict)

        # Key removed only in diff dict
        if "_name_or_path" in serializable_config_dict:
            del serializable_config_dict["_name_or_path"]

        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict) and self.quantization_config is not None
                else self.quantization_config
            )
        self.dict_dtype_to_str(serializable_config_dict)

        return serializable_config_dict

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PreTrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]

            output[key] = value

        self._remove_keys_not_serialized(output)

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict) and self.quantization_config is not None
                else self.quantization_config
            )
        self.dict_dtype_to_str(output)

        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PreTrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        # Handle +/-Infinity and NaNs
        config_dict = self._encode_special_floats(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike, use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PreTrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise TypeError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

    def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype") is not None:
            if isinstance(d["dtype"], dict):
                d["dtype"] = {k: str(v).split(".")[-1] for k, v in d["dtype"].items()}
            # models like Emu3 can have "dtype" as token in config's vocabulary map,
            # so we also exclude int type here to avoid error in this special case.
            elif not isinstance(d["dtype"], (str, int)):
                d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_dtype_to_str(value)

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        """
        Checks and removes if there are any keys in the dict that should not be serialized when saving the config.
        Runs recursive check on the dict, to remove from all sub configs.
        """

        if "_is_quantized" in d:
            del d["_is_quantized"]
        if "_auto_class" in d:
            del d["_auto_class"]
        if "_output_attentions" in d:
            d["output_attentions"] = d.pop("_output_attentions")
        if "_commit_hash" in d:
            del d["_commit_hash"]
        if "_attn_implementation_internal" in d:
            del d["_attn_implementation_internal"]
        if "_experts_implementation_internal" in d:
            del d["_experts_implementation_internal"]
        # Do not serialize `base_model_tp_plan` for now
        if "base_model_tp_plan" in d:
            del d["base_model_tp_plan"]
        # Do not serialize `base_model_pp_plan` for now
        if "base_model_pp_plan" in d:
            del d["base_model_pp_plan"]
        for value in d.values():
            if isinstance(value, dict):
                self._remove_keys_not_serialized(value)


def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.

    dict_b : the default config dictionary. We want to remove values that are in this one
    """
    diff = {}
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    for key, value in dict_a.items():
        obj_value = getattr(config_obj, str(key), None)
        if isinstance(obj_value, PreTrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            diff[key] = diff_value
        elif key not in dict_b or (value != default[key]):
            diff[key] = value
    return diff