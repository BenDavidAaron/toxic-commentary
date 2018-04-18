import pickle
from flask import Flask, render_template, request, redirect
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
app.vars = {}

app.vars['tokenizer'] = pickle.load(open('model-development/models/comment-tokenizer.pkl','rb'))
app.vars['model'] = load_model('model-development/models/cat-cross-model-e2.h5')
graph = tf.get_default_graph()
max_len = 200

lorem_ipsum = """Lorem ipsum dolor sit amet, nam cu rationibus honestatis instructior, sea esse numquam vulputate et, est cu velit tibique molestie. Cu quaeque percipit eum, ei sed quodsi blandit elaboraret. Iuvaret accusamus ei vis. Eos ne utamur commodo reformidans, noluisse menandri vel et, his conceptam sadipscing te.
Sed an hinc epicurei. Et sed fabulas scaevola senserit. Te ius errem vocent luptatum, ad putant viderer definiebas cum. Usu ipsum iudicabit ea, ne mei consul ancillae mentitum, an minimum invenire sed. Aeque choro ridens no quo, mei iudicabit temporibus te, sint gubergren consetetur usu ne.
Stet iisque propriae et sea. Duo malis recusabo suscipiantur at, te tempor graeco inermis mel, vim reque justo debet te. Ad qui dictas ponderum tincidunt, sed facer facete intellegat eu. Cu idque liber recteque eam, nec exerci everti invidunt ad. Ad qui sumo nonumes patrioque.
Vis ullum dolor scaevola ne, diam vocent an sea. Ea per congue qualisque honestatis, hinc quaeque adversarium vim no. An exerci principes vis. Ad atomorum inimicus disputando quo, te nec audiam feugiat reprehendunt. Sit consul prodesset moderatius ei, qui in noluisse voluptua laboramus. In duo elitr salutandi molestiae, nobis aliquam vulputate per ex. Ad mei iusto gubergren forensibus, ea choro vivendum praesent sea.
Congue quaeque officiis has id, dolor forensibus vel te, conceptam appellantur vel te. In iriure deseruisse signiferumque eam, ad sit possim quaeque. Id vix rebum populo molestiae, usu debitis nominavi no. Stet mazim evertitur has ne. Ne nec facer sadipscing. Vim semper nominati ex. Ea ius elit legere adipisci."""

@app.route('/', methods =['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', comment = lorem_ipsum, score = "")
    else:
        comment = str(request.form['comment'])
        global graph
        with graph.as_default():
            #do inference here
            tokens = app.vars['tokenizer'].texts_to_sequences([comment])
            arr = pad_sequences(tokens, maxlen=max_len)
            pred = app.vars['model'].predict(arr)[0][1]
            tox_score = f'We\'re {pred*100:{5}.{5}}% certain that this comment is Toxic'
        return render_template('index.html', comment=comment, score = tox_score)
if __name__ == '__main__':
    app.run(port=33507, debug=True)
