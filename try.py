import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the tokenizer and model
tokenizer = BartTokenizer.from_pretrained('tweet_reply_model')
model = BartForConditionalGeneration.from_pretrained('tweet_reply_model')
model.load_state_dict(torch.load("tweet_reply_model.pt"))
model.eval()

# Function to generate a reply
def generate_reply(tweet, max_length=50):
    input_ids = tokenizer.encode(tweet, return_tensors='pt')

    # Generate a reply
    reply_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Example input
example_tweet = "I got my reserach paper published today."

# Generate a reply
reply = generate_reply(example_tweet)
print(f"Tweet: {example_tweet}")
print(f"Reply: {reply}")
