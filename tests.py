####################
# PLACE TESTS HERE #
def test_read_data_data_types():
    data = read_data("UD_English-GUM/en_gum-ud-train.conllu")
    result = {
        'is_data_list': type(data) == list,
        'is_data_first_element_list': type(data[0]) == list,
        'is_data_first_element_first_item_tuple': type(data[0][0]) == tuple
    }
    return result

def test_read_data_len_train_data():
    return {
        'train_data_length': len(read_data("UD_English-GUM/en_gum-ud-train.conllu")),
    }

def test_generate_vocabs():
    return {
        'vocab_size': len(words_vocab),
        'num_tags': len(tags_vocab)
    }

def test_hmm():
    return {
        'precision': round(precision_hmm, 2),
        'recall': round(recall_hmm, 2),
        'f1': round(f1_hmm, 2),
    }

def test_embeddings_model():
    return {
        'precision': round(precision_embeddings, 2),
        'recall': round(recall_embeddings, 2),
        'f1': round(f1_embeddings, 2),
    }
    

def test_nltk_tagger():
    return {
        'accuracy': round(accuracy_tnt_pos_tagger, 2)
    }

TESTS = [test_read_data_data_types, test_read_data_len_train_data, test_generate_vocabs, test_hmm, test_embeddings_model, test_nltk_tagger]

# Run tests and save results
res = {}
for test in TESTS:
    try:
        cur_res = test()
        res.update({test.__name__: cur_res})
    except Exception as e:
        res.update({test.__name__: repr(e)})

with open('results.json', 'w') as f:
    json.dump(res, f, indent=2)

# Download the results.json file
files.download('results.json')

####################
