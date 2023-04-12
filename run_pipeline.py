import os
import subprocess
import pandas as pd
from pmaw import PushshiftAPI
import praw
import re
import datetime as dt
import transformers
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tqdm as notebook_tqdm
import nltk
import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
import spacy
from collections import Counter
import requests
import json


class RedditPull:
    def __init__(self, medication):
        self.medication = medication

    def reddit_pull(self):
        reddit = praw.Reddit(
            client_id=os.environ['REDDIT_CLIENT_ID'],
            client_secret=os.environ['REDDIT_CLIENT_SECRET'],
            user_agent='test-app'
        )

        api_praw = PushshiftAPI(praw=reddit)

        year = 2022  # Set the year you want to iterate through
        all_comments = pd.DataFrame()

        for month in range(1, 13):  # Iterate through each month
            start_date = int(dt.datetime(year, month, 1).timestamp())

            if month == 12:
                end_date = int(dt.datetime(year + 1, 1, 1).timestamp())
            else:
                end_date = int(dt.datetime(year, month + 1, 1).timestamp())

            comments = api_praw.search_comments(
                q=self.medication,
                limit=100,
                after=start_date,
                before=end_date
            )

            print('Retrieved {} comments for {}-{}'.format(len(comments), year, month))
            month_comments = pd.DataFrame(comments)
            all_comments = pd.concat([all_comments, month_comments], ignore_index=True)

        print('Retrieved total {} comments'.format(len(all_comments)))

        return all_comments

    
    def preprocess_text(self,text):
        # remove links
        text = re.sub(r'http\S+', '', text)
        # remove numbers
        text = re.sub(r'\d+', '', text)
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    #Function to split the comments into lists of strings no more than 200 tokens and no less than 150 tokens long
    def get_split(self,text1):
        l_total = []
        l_partial = []
        if len(text1.split())//150 >0:
            n = len(text1.split())//150
        else: 
            n = 1
        for w in range(n):
            if w == 0:
                l_partial = text1.split()[:200]
                l_total.append(" ".join(l_partial))
            else:
                l_partial = text1.split()[w*150:w*150 + 200]
                l_total.append(" ".join(l_partial))
        return l_total
        
    def create_dataframe(self, comments):
        if comments is None:
            print('Cannot create dataframe with empty comments')
            return
        
        reddit_data = comments[['body']].copy()
        
        #Apply preprocessing and splitting
        reddit_data['process_text'] = reddit_data['body'].apply(self.preprocess_text)
        reddit_data['process_text'] = reddit_data['process_text'].apply(self.get_split)
        
        #Set the target column because the data loaders and stuff expect it to be there
        reddit_data['target'] = 0
        return reddit_data

class RedditCommentDataset(Dataset):
    
    def __init__(self, doc, targets, tokenizer, max_len):
        self.doc = doc
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.doc)
  
    def __getitem__(self, item):
        doc = str(self.doc[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                    )

        return {
            'doc_text': doc,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
            }
            
class ADRClassifier():

    def __init__(self, input_df, model_path, device=None):
        self.RANDOM_SEED = 42
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.load_model(model_path).to(self.device)
    
    @staticmethod
    def load_model(model_path):
        model = torch.load(model_path)
        return model
            
    #Function to create Pytorch data loaders using previously built Dataset
    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = RedditCommentDataset(
            doc=df.process_text.to_numpy(),
            targets=df.target.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,
        )

            
    #Get the predictions
    def get_predictions(self, model, data_loader):
        model = model.eval()

        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:

                texts = d["doc_text"]
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                probs = F.softmax(outputs, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values

        
    def main(self, input_df):
        
        model_path = "models/roberta-classifier/best_model_with_gpt.pt"
        # Check if GPU is available and use it, otherwise use CPU
        if torch.cuda.is_available():
            model = self.load_model(model_path).to('cuda')
            print(f"Model loaded onto {torch.cuda.get_device_name(0)}")
        else:
            model = self.load_model(model_path).to('cpu')
            print("Model loaded onto CPU")       

        MAX_LEN = 128
        BATCH_SIZE = 16

        n_classes = 2
        tokenizer=AutoTokenizer.from_pretrained('roberta-base',use_fast=False)

        # Instantiate data loader
        data_loader = self.create_data_loader(input_df, tokenizer, MAX_LEN, BATCH_SIZE)

        # Get the comment texts, the predicted classes, the predicted class probabilities, and the real values
        # Real values obviously don't matter for predictions on newly scraped unlabeled data
        y_review_texts, y_pred, y_pred_probs, y_test = self.get_predictions(model, data_loader)

        
        #Create dataframe for results with comment text and accompanying predicted classes
        y_pred_numpy = y_pred.numpy()
        texts_df = pd.DataFrame(y_review_texts, columns=['text'], index=None)
        preds_df = pd.DataFrame(y_pred_numpy,columns=['preds'], index=None)
        results_df = pd.concat([texts_df, preds_df], axis=1)
        
        #Create dataframe for just the comments classified as ADRs so they can be moved to the next step in the pipeline
        classified_adrs = results_df.loc[results_df['preds'] == 1]
        adr_text = classified_adrs[['text']].copy()
        # Save updated data to new CSV file
        
        return adr_text
      
class PostClassifier(nn.Module):
    def __init__(self, n_classes):
        super(PostClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained('roberta-base', return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

class ADRLabeler:
    def __init__(self, input_df, model_path):
        self.model = self.load_model(model_path).to('cuda')
        print(f"Model loaded onto {torch.cuda.get_device_name(0)}")
        self.drug_labeled = self.process_input_dataframe(input_df)

    @staticmethod
    def load_model(model_path):
        model = SequenceTagger.load(model_path)
        return model

    def predict(self, text):
        try:
            sentence = Sentence(text)
            self.model.predict(sentence)
            return sentence.to_tagged_string()
        except Exception as e:
            print(f"Error predicting text: {e}")
            return ""

    def process_input_dataframe(self, input_df):
        # Add new column with labeled ADRs
        input_df['label'] = input_df['text'].apply(lambda x: self.predict(x))
        return input_df

    def main(self, input_df):
        model_path = "models/flair-ner/best-model.pt"
        predictor = TextPredictor(model_path)
        predictor.process_input_dataframe(input_df)

class ADRLinker:
    def __init__(self, input_df, model_path):
        self.nlp = spacy.load("en_core_web_md")
        self.model = SequenceTagger.load(model_path).to('cuda')
        print(f"Model loaded onto {torch.cuda.get_device_name(0)}")
        self.drug_labeled = input_df
        self.results_df = None

    def get_shortest_dependency_path(self, start_token, end_token):
        """
        Finds the shortest path in the dependency tree between two tokens
        """
        path = list(start_token.ancestors) + [start_token] + list(end_token.ancestors)
        path = [token for token in path if token.is_alpha]
        if len(path) == 0:
            return None
        path.sort(key=lambda token: token.i)
        return [token.text for token in path]

    def pair_drugs_adrs_dependency(self, text, ner_labels):

        pairs = []

        for i, label in enumerate(ner_labels):
            entity_text, label_type = label.split('/')
            start_pos = text.find(entity_text)
            end_pos = start_pos + len(entity_text)
            entity_span = self.nlp(text).char_span(start_pos, end_pos, label=label_type)

            if entity_span is not None:
                for j in range(i + 1, len(ner_labels)):
                    other_entity_text, other_label_type = ner_labels[j].split('/')

                    # Ensure that one label is a drug and the other is an ADR
                    if (label_type == "DRUG" and other_label_type == "ADR") or (label_type == "ADR" and other_label_type == "DRUG"):
                        other_start_pos = text.find(other_entity_text)
                        other_end_pos = other_start_pos + len(other_entity_text)
                        other_entity_span = self.nlp(text).char_span(other_start_pos, other_end_pos, label=other_label_type)

                        if other_entity_span is not None:
                            path = self.get_shortest_dependency_path(entity_span[0], other_entity_span[0])

                            if path:
                                if label_type == "DRUG":
                                    pairs.append({"drug": entity_text.lower(), "adr": other_entity_text.lower()})
                                else:
                                    pairs.append({"drug": other_entity_text.lower(), "adr": entity_text.lower()})
        return pairs


    def extract(self, output_path):
        
        pairs_list = []
        
        for text in self.drug_labeled['text']:         
            sentence_obj = Sentence(text)
            self.model.predict(sentence_obj)
            ner_labels = [f"{entity.text}/{entity.labels[0].value}" for entity in sentence_obj.get_spans('ner')]
            pairs = self.pair_drugs_adrs_dependency(text, ner_labels)
            pairs_list.append(pairs)   
                          
        self.drug_labeled['drug_adr_pairs'] = pairs_list

        # create an empty dictionary to store the drug-ADR pairs
        drug_adr_dict = {}

        # loop over each row in the 'drug_adr_pairs' column
        for pairs in self.drug_labeled['drug_adr_pairs']:
            # update the drug-ADR dictionary with the new pairs
            for pair in pairs:
                drug = pair['drug']
                adr = pair['adr']
                
                if drug not in drug_adr_dict:
                    drug_adr_dict[drug] = []
                
                drug_adr_dict[drug].append(adr)

        # create a list of dictionaries, where each dictionary corresponds to a drug and its ADRs, along with the number of occurrences
        results = []

        for drug, adrs in drug_adr_dict.items():
            count = Counter(adrs)
            adr_counts = count.most_common()
            adr_list = [{'adr': adr, 'occurrences': count} for adr, count in adr_counts]
            results.append({'drug': drug, 'adrs': adr_list, 'total_occurrences': sum(count.values())})

        # create a dataframe from the results list and sort it by the number of total occurrences in descending order
        self.results_df = pd.DataFrame(results).sort_values(by='total_occurrences', ascending=False)
        self.results_df.to_csv(output_path, index=False)
        print(f"Saved output to {output_path}.")

def get_faers(medication):
    
    print(f"Fetching FAERS data for: {medication}")
    
    print(f"Medication variable type: {type(medication)}")

    accepted = False
    try:
        response = requests.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.brand_name:'+medication+'&limit=20&count=patient.reaction.reactionmeddrapt.exact')
        data = response.json()
        df = pd.DataFrame(data['results'])
        accepted = True
    except:
        accepted = False

    if accepted == False:
        try:
            response = requests.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:'+medication+'&limit=20&count=patient.reaction.reactionmeddrapt.exact')
            data = response.json()
            df = pd.DataFrame(data['results'])
            accepted = True
        except:
            accepted = False
    if accepted == False:
        print(f"{medication} invalid")
    else:
        df.to_csv('shiny_app/faers.csv')

def run_pipeline(medication):

    print("Starting pipeline for medication:", medication)
    reddit_pull = RedditPull(medication)
    get_faers(medication)

    print("Pulling comments from Reddit...")
    comments = reddit_pull.reddit_pull()

    print("Finished pulling comments. Creating dataframe...")
    reddit_data = reddit_pull.create_dataframe(comments)
 
    print("Finished creating dataframe. Classifying texts as containing an adverse drug reaction (ADR) or not...")
    adr_classifier = ADRClassifier(input_df=reddit_data, model_path="models/roberta-classifier/best_model_with_gpt.pt")
    classified_data = adr_classifier.main(reddit_data)  

    print("Finished classifying texts. Labeling drug name and words or phrases indicating an ADR...")
    adr_labeler = ADRLabeler(input_df=classified_data, model_path="models/flair-ner/best-model.pt")
    labeled_data = adr_labeler.drug_labeled

    print("Finished labeling. Linking drugs and ADRs syntactically...")
    linked_data = ADRLinker(input_df=labeled_data, model_path="models/flair-ner/best-model.pt")
    linked_data.extract(output_path="shiny_app/linked_data.csv")

    print("Finished linking drugs and ADRs.")

if __name__ == '__main__':
    
    medication = os.getenv("MEDICATION_NAME") 
    
    run_pipeline(medication)

