import dash
import dash_html_components as html
import dash_core_components as dcc
import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer


def predict(reviews, model):
    with torch.no_grad():
        model.eval()

        input_id, attention_masks = preprocess(reviews)
        input_id = input_id
        attention_masks = attention_masks

        retour = model(input_id, attention_mask=attention_masks)

        return torch.argmax(retour[0], dim=1)


def preprocess(raw_reviews, sentiments=None):
    encoded_batch = TOKENIZER.batch_encode_plus(raw_reviews,
                                                add_special_tokens=True,
                                                max_length=None,
                                                padding=True,
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
    if sentiments:
        sentiments = torch.tensor(sentiments)
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], sentiments
    return encoded_batch['input_ids'], encoded_batch['attention_mask']





external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    }
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(
    children=[

        html.Div(
            children=[
                html.P(children="🎬", className="header-emoji"),
                html.H1(
                    children="Movie reviews sentiment classification", className="header-title"
                ),
                html.P(
                    children="Sentiment classification of movie reviews."
                             "\n Made using CamemBERT and trained on Allociné reviews."
                             "\n Type a movie reviews and then submit !",
                    className="header-description",
                ),
            ],
            className="header",
        ),

        html.Div(
            children=[
                html.Div(dcc.Input(id='input-on-submit', type='text')),
                html.Button('Submit', id='submit-val', n_clicks=0),

            ], style={'width': '100%', 'display': 'flex', 'flex-direction': 'row', 'align-items': 'center',
                      'justify-content': 'center', 'marginTop': '50px', 'marginBottom': '50px'}
        ),

        html.Div(
            children=[

                html.Div(id='container-button-basic', children='Type a review and press submit'),
            ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center',
                      'justifyContent': 'center'}
        ),
    ]
)


@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    if value is None or "":
        return 'Please type a review'

    if predict([str(value)], model) == torch.Tensor([1]):
        return 'The  review submitted is considered Positive'
    else:
        return 'The review submitted is considered Negative'


if __name__ == '__main__':
    TOKENIZER = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
    model = CamembertForSequenceClassification.from_pretrained(
        'camembert-base',
        num_labels=2)
    model.load_state_dict(torch.load("enter here the url of you model you trained before using training.py"))

    print("model loaded")
    app.run_server(debug=True)
