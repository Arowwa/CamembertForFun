import torch
import pandas as pd
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

# Chargement du jeu de donnees

dataset = pd.read_csv("reviews_allocine_classifications.csv")


reviews = dataset['text'].values.tolist()
# print(reviews)
sentiments = dataset['label'].values.tolist()
print(len(sentiments))

# On charge l'objet 'tokenizer' de camembert qui va servir a encoder
# 'camembert-base' est la version de camembert qu'on a choisi d'utiliser
# 'do_lower_case' à True pour qu'on passe tout en minuscule
TOKENIZER = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)


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


# split train/validation
# On utilise 80% du jeu de donnée pour l'entrainement et les 20% restant pour la validation
split_border = int(len(sentiments) * 0.8)

reviews_train, reviews_validation = reviews[:split_border], reviews[split_border:]
sentiments_train, sentiments_validation = sentiments[:split_border], sentiments[split_border:]

input_ids, attention_mask, sentiments_train = preprocess(reviews_train, sentiments_train)

train_dataset = TensorDataset(
    input_ids,
    attention_mask,
    sentiments_train)

input_ids, attention_mask, sentiments_validation = preprocess(reviews_validation, sentiments_validation)

validation_dataset = TensorDataset(
    input_ids,
    attention_mask,
    sentiments_validation)

batch_size = 2

# On crée les DataLoaders d'entrainement et de validation
# Le DataLoader est juste un objet iterable
# on le configure pour itérer le jeu d'entrainement de façon aleatoire et creer les batchs.

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size)
validation_dataloader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    batch_size=batch_size)

# On charge la version pre-entrainee de camemBERT 'base'
model = CamembertForSequenceClassification.from_pretrained(
    'camembert-base',
    num_labels=2)

model = model.to(device)

# On utilise l'Adam optimizer avec les paramètres par défaut et on entraine sur 3 epoques
optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8)
epochs = 5

# Pour enregistrer les stats a chaque epoque
training_stats = []


# Boucle d'entrainement
def training_model(train_dataloader, model):
    for epoch in range(0, epochs):

        print("")
        print(f'######### Epoch {epoch + 1} / {epochs} ###########')
        print('Training...')

        # On initialise la loss pour cette epoch
        total_train_loss = 0

        # On met le modele en mode training
        # Dans ce mode certaines couches du modèle agissent differement
        model.train()

        for step, batch in enumerate(train_dataloader):

            # On fait un print chaque 40 batch
            if step % 40 == 0 and not step == 0:
                print(f'Batch {step} of {len(train_dataloader)}.')

            # On recupere les donnees du batch
            input_id = batch[0].to(device)
            attention_mask = batch[1].to(device)
            sentiment = batch[2].to(device)

            # On met le gradient a 0
            model.zero_grad()

            # On passe la donnee au model et on recupere la loss et le logits (sortie avant la fonction d'activation)
            loss, logits = model(input_id,
                                 token_type_ids=None,
                                 attention_mask=attention_mask,
                                 labels=sentiment)

            # On incremente la loss totale
            # .item() donne la valeur numérique de la loss
            total_train_loss += loss.item()

            # Backprop
            loss.backward()

            # On actualise les params grace a l'optimizer
            optimizer.step()

        # On calcule la loss moyenne sur toute l'epoque
        avg_train_loss = total_train_loss / len(train_dataloader)

        print("")
        print("   Average training loss : {0:.2f}".format(avg_train_loss))

        # Enregistrement des stats de l'epoque
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
            }
        )
    print('Model saved!')
    torch.save(model.state_dict(), "./sentimentClassification.pt")


def predict(reviews, model=model):
    with torch.no_grad():
        model.eval()

        input_id, attention_masks = preprocess(reviews)
        input_id = input_id.to(device)
        attention_masks = attention_masks.to(device)

        retour = model(input_id, attention_mask=attention_masks)

        return torch.argmax(retour[0], dim=1).cpu()


def evaluate(reviews, sentiments, metric='report'):
    predictions = predict(reviews)
    print(predictions)
    if metric == 'report':
        return metrics.classification_report(sentiments, predictions, zero_division=0)
    elif metric == 'matrix':
        return metrics.confusion_matrix(sentiments, predictions)


training_model(train_dataloader, model)
confusion_matrix = evaluate(reviews_validation, sentiments_validation, 'matrix')
report = evaluate(reviews_validation, sentiments_validation, 'report')
print(report)

# Pour recharger le model un fois entraine
# model.load_state_dict(torch.load("./sentimentClassification.pt"))