"""Load the pinned AST offline and shorten its existing positional embeddings."""
import json
import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification


def loading_metadata(info):
    return json.loads(json.dumps(info, default=lambda value: sorted(value) if isinstance(value, set) else str(value)))


def crop_positions(model, max_length):
    embeddings = model.audio_spectrogram_transformer.embeddings
    config = model.config
    frequencies, old_time = embeddings.get_shape(config)
    new_time = (max_length - config.patch_size) // config.time_stride + 1
    if not 0 < new_time <= old_time:
        raise ValueError('Only cropping to a positive shorter time axis is supported')
    positions = embeddings.position_embeddings
    grid = positions[:, 2:].reshape(1, frequencies, old_time, config.hidden_size)
    cropped = grid[:, :, :new_time, :].reshape(1, frequencies * new_time, config.hidden_size)
    embeddings.position_embeddings = torch.nn.Parameter(torch.cat([positions[:, :2], cropped], dim=1).detach().clone())
    config.max_length = max_length
    embeddings.config.max_length = max_length
    return dict(frequency_patches=frequencies, old_time_patches=old_time,
                time_patches=new_time, tokens=frequencies * new_time + 2)


def create_ast(config, labels):
    extractor = ASTFeatureExtractor.from_pretrained(config['model_id'], revision=config['revision'],
        local_files_only=config['local_files_only'], max_length=config['max_length'])
    model_config = ASTConfig.from_pretrained(config['model_id'], revision=config['revision'],
                                             local_files_only=config['local_files_only'])
    model_config.num_labels = config['num_classes']
    model_config.id2label = labels
    model_config.label2id = {label: index for index, label in labels.items()}
    model_config.problem_type = 'single_label_classification'
    model, loading = ASTForAudioClassification.from_pretrained(
        config['model_id'], revision=config['revision'], local_files_only=config['local_files_only'],
        config=model_config, ignore_mismatched_sizes=True,
        attn_implementation=config['attention_implementation'], output_loading_info=True)
    layout = crop_positions(model, config['max_length'])
    return model, extractor, layout, loading_metadata(loading)
