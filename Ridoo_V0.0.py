#!/usr/bin/env python
# coding: utf-8

# In[15]:


#INstall the transformers library
get_ipython().system('pip install transformers')


# In[10]:


#Declare a seed value for better reproducability
SEED = 4243    
#permet d'assurer la reproductibilité des résultats en fixant une valeur de départ pour l'initialisation aléatoire


# In[12]:


get_ipython().system('pip list | findstr /R "torch nvidia fsspec gcsfs"')



# In[13]:


#Install the datasets library
get_ipython().system('pip install datasets')
#est utilisée pour télécharger, prétraiter et manipuler des jeux de données de machine learning de manière efficace.


# In[18]:


get_ipython().system('pip install datasets')


# In[21]:


from datasets import load_dataset

# Charger le dataset "kde4" en anglais-français avec l'option trust_remote_code
raw_datasets = load_dataset("kde4", "en-fr", trust_remote_code=True)

# Afficher les premières lignes du dataset
print(raw_datasets)


# In[22]:


# Afficher quelques exemples
print(raw_datasets["train"][0])  # Afficher la première ligne du dataset


# In[23]:


# Afficher quelques exemples
print(raw_datasets["train"][120])  # Afficher la première ligne du dataset


# In[24]:


get_ipython().system('pip install transformers')


# In[25]:


from datasets import load_dataset

# Définir une graine pour la reproductibilité
SEED = 4243

# Charger le dataset KDE4 en anglais-français
raw_datasets = load_dataset("kde4", "en-fr", trust_remote_code=True)

# Mélanger les données du jeu d’entraînement et sélectionner 5 échantillons
random_samples = raw_datasets["train"].shuffle(seed=SEED).select(range(5))

# Afficher les échantillons sélectionnés
for sample in random_samples:
    print(sample)
    print("\n\t\t\t\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")


# In[26]:


from datasets import load_dataset

# Définir une graine pour la reproductibilité
SEED = 42

# Charger le dataset KDE4 en anglais-français
raw_datasets = load_dataset("kde4", "en-fr", trust_remote_code=True)

# Diviser le dataset en 80% pour l'entraînement et 20% pour le test
split_datasets = raw_datasets["train"].train_test_split(train_size=0.8, seed=SEED)

# Afficher la structure des ensembles créés
print(split_datasets)


# In[1]:


print("Hi Siwar")


# In[2]:


#Install the sacremoses library
get_ipython().system('pip install sacremoses')
# un outil de tokenization et de prétraitement très utilisé en traduction automatique.


# In[3]:


get_ipython().system('pip install tf-keras')


# In[7]:


get_ipython().system('pip install tokenizers')

Data Preprocessing
# In[1]:


print("Hi Siwar")


# In[4]:


get_ipython().system('pip install torch torchvision torchaudio')


# In[2]:


print("Hi Siwar")


# In[4]:


get_ipython().system('pip install sentencepiece')


# In[5]:


import importlib.util

package_name = "sentencepiece"

if importlib.util.find_spec(package_name) is not None:
    print(f"{package_name} est installé ✅")
else:
    print(f"{package_name} n'est PAS installé ❌")


# In[10]:


import sentencepiece
print(sentencepiece.__version__)


# In[6]:


print("Hi Siwar")


# In[7]:


from transformers import pipeline

# Load a translation pipeline as a test
translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-fr")

# Test translation
result = translator("Hello, how are you today?")
print(result)


# In[8]:


from transformers import AutoTokenizer

# Define the checkpoint or model path
checkpoint = "Helsinki-NLP/opus-mt-en-fr"

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="tf")  # return_tensors="tf" ensures TensorFlow tensors are returned

# Example usage: Tokenizing a sample sentence
input_text = "Hello, how are you today?"
encoded_input = tokenizer(input_text, return_tensors="tf")

# Check the tokenized output
print(encoded_input)


# In[13]:


import sys
print(sys.executable)


# In[14]:


get_ipython().system('pip install datasets')


# In[6]:


get_ipython().system('pip install --upgrade datasets huggingface_hub')


# In[7]:


from datasets import load_dataset, DownloadConfig

# Create a DownloadConfig object with options
download_config = DownloadConfig()

# Load the IMDb dataset with the specified download config
dataset = load_dataset("imdb", download_config=download_config)

# Print the dataset to see the result
print(dataset)


# In[8]:


from datasets import load_dataset

# Load the 'wmt14' dataset with the 'fr-en' configuration (French-English translation)
dataset = load_dataset("wmt14", "fr-en")

# Access the second example in the 'train' split
sample = dataset["train"][1]

# Display the sample
print(sample)


# In[10]:


from datasets import load_dataset

# Load the 'wmt14' dataset (example: translation dataset)
dataset = load_dataset("wmt14", "fr-en")

# Extract the English and French sentences for the sample at index 20
i = 20
en_sentence = dataset["train"][i]["translation"]["en"]
fr_sentence = dataset["train"][i]["translation"]["fr"]

# Display the English and French sentences
print("English:", en_sentence)
print("French:", fr_sentence)


# In[11]:


from transformers import AutoTokenizer

# Load the tokenizer for the pre-trained model you are using
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")  # Example for English to French translation

# Example sentences (replace with the ones you've extracted)
en_sentence = "That is precisely the time when you may, if you wish, raise this question, i.e. on Thursday prior to the start of the presentation of the report."
fr_sentence = "C'est exactement à ce moment-là que vous pourrez, en effet, si vous le souhaitez, soulever cette question, c'est-à-dire jeudi avant le début de la présentation du rapport."

# Tokenize the sentences
model_input = tokenizer(en_sentence, text_target=fr_sentence)

# Print the tokenized output
print(model_input)


# In[13]:


get_ipython().system('pip install sacremoses')


# In[14]:


from transformers import AutoTokenizer

# Load the tokenizer for the pre-trained model you are using
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")  # Example for English to French translation

# Example sentences (replace with the ones you've extracted)
en_sentence = "That is precisely the time when you may, if you wish, raise this question, i.e. on Thursday prior to the start of the presentation of the report."
fr_sentence = "C'est exactement à ce moment-là que vous pourrez, en effet, si vous le souhaitez, soulever cette question, c'est-à-dire jeudi avant le début de la présentation du rapport."

# Tokenize the sentences
model_input = tokenizer(en_sentence, text_target=fr_sentence)

# Print the tokenized output
print(model_input)


# In[16]:


#Let's convert these tokens back to words to
# check how the tokenizer performed

#English (source language) word tokens
print("English Tokens:",tokenizer.convert_ids_to_tokens(model_input["input_ids"]))

#French (target language) word tokens
print("French Tokens:",tokenizer.convert_ids_to_tokens(model_input["labels"]))  #convertir les identifiants de tokens en mots (ou sous-mots) pour vérifier
                                                                                #la performance du tokenizer en termes de segmentation des phrases source et cible


# In[18]:


max_length = 128

def tokenize_dataset(examples):
    #Extract the English sentence from the given sample
    inputs = [ex["en"] for ex in examples["translation"]]
    #Extract the French sentence from the given sample
    targets = [ex["fr"] for ex in examples["translation"]]

    #Apply tokenizer
    model_inputs = tokenizer(inputs,
                             text_target=targets,
                             max_length=max_length,
                             truncation=True
                            )
    return model_inputs      #Cette fonction permet de préparer un ensemble de données de traduction pour l'entraînement
                             #en tokenisant les phrases en anglais et en français, tout en s'assurant qu'elles respectent une longueur maximale.


# In[23]:


get_ipython().system('pip install --upgrade transformers')


# In[24]:


from transformers import MBartTokenizer


# In[1]:


from datasets import load_dataset
from transformers import MBartTokenizer

# Load the dataset (e.g., 'wmt14' translation dataset)
split_datasets = load_dataset("wmt14", "fr-en")

# Initialize the tokenizer
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")

# Check the structure of the examples
def tokenize_dataset(examples):
    print(examples)  # This will show the structure of the data
    return tokenizer(examples['en'], text_target=examples['fr'], padding="max_length", truncation=True)

# Apply the tokenization to the dataset
tokenized_dataset = split_datasets.map(function=tokenize_dataset, batched=True, remove_columns=split_datasets["train"].column_names)

# Print the tokenized dataset
print(tokenized_dataset)


# In[2]:


print(examples)


# In[3]:


return tokenizer(examples['translation_en'], text_target=examples['translation_fr'], padding="max_length", truncation=True)


# In[4]:


from datasets import load_dataset

# Load the dataset (example: wmt14)
split_datasets = load_dataset("wmt14", "fr-en")

# Check the structure of the first example in the training set
print(split_datasets['train'][0])


# Model

# In[7]:


from transformers import TFAutoModelForSeq2SeqLM

# Step 1: Define the checkpoint (pre-trained model identifier)
checkpoint = "t5-small"  # Example model (you can use any pre-trained model)

# Step 2: Load the pre-trained model
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)


# Data collator

# In[9]:


from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,return_tensors="tf") # Ce code prépare automatiquement les données (padding, batchs)
                                                                                         #avant  de les passer au modèle de traduction pour l'entraînement


# In[14]:


get_ipython().system('pip install torch')


# In[16]:


from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

# Load a tokenizer (e.g., for T5)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Load the dataset (e.g., 'imdb' dataset, adjust according to your task)
dataset = load_dataset("imdb")

# Tokenize the dataset (tokenize the "train" split here)
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True), batched=True)

# Ensure the 'labels


# In[24]:


from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

# Load a tokenizer (e.g., for T5)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Load the dataset (e.g., 'imdb' dataset, adjust according to your task)
dataset = load_dataset("imdb")

# Tokenize the dataset (tokenize the "train" split here)
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True), batched=True)

# Ensure the 'labels' field is correctly set
# For sequence-to-sequence, labels are typically shifted. Here we use input_ids as labels directly.
# Adjust for seq2seq needs; typically, labels are shifted for autoregressive models
tokenized_dataset = tokenized_dataset.map(lambda x: {'labels': x['input_ids']}, batched=True)

# Inspect the first element to ensure it has the correct structure
print(tokenized_dataset["train"][0])  # Print the first entry for inspection

# Initialize the data collator for TensorFlow
data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)

# Test the data collator on the first 5 samples of the "train" split
try:
    collated_samples = data_collator([sample for sample in tokenized_dataset["train"].select(range(5))], return_tensors="tf")

    # Convert English tokens back to words (for the input)
    print("English Tokens back to Words:")
    print(tokenizer.convert_ids_to_tokens(collated_samples["input_ids"][2]))

    print("\n\t\t\t\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    # Convert French tokens back to words (for the labels)
    print("French Tokens back to Words:")
    print(tokenizer.convert_ids_to_tokens(collated_samples["labels"][2]))
except Exception as e:
    print(f"Error: {e}")


# In[41]:


get_ipython().system('pip install tensorflow')


# In[42]:


from transformers import AutoTokenizer, TFT5ForConditionalGeneration

# Load the tokenizer and model (ensure TensorFlow version)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = TFT5ForConditionalGeneration.from_pretrained("t5-small")  # This is a TensorFlow model

# Test the model with an example input text
input_text = "Translate English to French: How are you?"
inputs = tokenizer(input_text, return_tensors="tf")

# Generate an output
outputs = model.generate(inputs['input_ids'], max_new_tokens=50)

# Decode the generated output and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")


# Model fine-tuning

# In[50]:


import tensorflow as tf
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback

# Exemple fictif de données d'entraînement et d'étiquettes (remplacez-les par vos propres données)
train_features = tf.random.normal([1000, 128])  # 1000 exemples, chaque exemple ayant 128 caractéristiques
train_labels = tf.random.uniform([1000], maxval=2, dtype=tf.int32)  # 1000 étiquettes (0 ou 1 pour une classification binaire)

# Créer le dataset TensorFlow
tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))

# Configurer le batch size
batch_size = 32
tf_train_dataset = tf_train_dataset.batch(batch_size)

# Paramètres d'entraînement
num_epochs = 5
num_train_steps = len(train_features) // batch_size * num_epochs  # Calculer le nombre d'étapes d'entraînement

# Créer l'optimiseur
optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)

# Modèle fictif (remplacez-le par votre modèle réel)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Classification binaire
])

# Compiler le modèle
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(tf_train_dataset, epochs=num_epochs)


# In[54]:


#Log in to HuggingFace account
from huggingface_hub import notebook_login

notebook_login()


# In[ ]:




