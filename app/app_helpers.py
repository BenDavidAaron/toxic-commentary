tokenizer = pickle.load(open('model-development/models/comment-tokenizer.pkl','rb'))
model = load_model('model-development/models/cat-cross-model-e2.h5')
max_len = 200

def Score_Comment(comment, model, tokenizer, expected_len):
    tokens = tokenizer.texts_to_sequences([comment])
    arr = pad_sequences(tokens, maxlen=expected_len)
    pred = model.predict(arr)[0]
    return float(pred[1])

