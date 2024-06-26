import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd

# Define dataset class
class TweetReplyDataset(Dataset):
    def __init__(self, tweets, replies, tokenizer, max_len):
        self.tweets = tweets
        self.replies = replies
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        reply = str(self.replies[item])

        source = self.tokenizer.encode_plus(
            tweet,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        target = self.tokenizer.encode_plus(
            reply,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': source['input_ids'].flatten(),
            'attention_mask': source['attention_mask'].flatten(),
            'labels': target['input_ids'].flatten(),
            'reply_text': reply  # Add actual reply text to the batch
        }

# Load data
data = pd.read_csv('/all/cse/uday/WIDS/dataa.csv')  # replace with your dataset path
tweets = data['Tweet'].values
replies = data['Reply'].values

# Initialize tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Define max length
MAX_LEN = 128

# Create dataset
dataset = TweetReplyDataset(tweets, replies, tokenizer, MAX_LEN)

# Create data loader
BATCH_SIZE = 4
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = BartForConditionalGeneration.from_pretrained('tweet_reply_model')  # replace with your model directory

# Evaluate accuracy
total_correct = 0
total_samples = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

with torch.no_grad():
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Generate replies
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LEN, num_beams=4, length_penalty=2.0, early_stopping=True)
        generated_replies = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Calculate accuracy
        for gen_reply, true_reply in zip(generated_replies, batch['reply_text']):
            if gen_reply == true_reply:
                total_correct += 1
            total_samples += 1

# Calculate accuracy
accuracy = total_correct / total_samples

print(f'Accuracy: {accuracy:.4f}')
