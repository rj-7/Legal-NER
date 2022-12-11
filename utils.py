import json
import spacy
from spacy.training import offsets_to_biluo_tags
from spacy.training import biluo_to_iob


def load_training_data(filepath: str, is_preamble: bool):
    f = open(filepath)
    data = json.load(f)
    f.close()
    sample_list = list()
    for sentence_obj in data:
        data_obj = sentence_obj['data']
        sample_text = data_obj['text']
        annotations = sentence_obj['annotations'][0]
        annotations = annotations['result']
        entity_offsets = list()
        for annotation in annotations:
            entity = annotation['value']
            entity_offsets.append((entity['start'], entity['end'], entity['labels'][0]))
        sample_list.append((sample_text, is_preamble, entity_offsets))             
    return sample_list



def convert_biluo_scheme(data, nlp):
    biluo_labels = list()
    bio_labels = list()
    for sentence, _, entities in data:
        doc = nlp(sentence)
        biluo_tags = offsets_to_biluo_tags(doc, entities)
        bio_tags = biluo_to_iob(biluo_tags)
        biluo_labels.append(biluo_tags)
        bio_labels.append(bio_tags)
    return biluo_labels, bio_labels

def encode_label_ids(labels):
    unique_labels = set(labels)
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    return labels_to_ids, ids_to_labels