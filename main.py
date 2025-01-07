import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
# Loading Hugging Face Finbert model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

#function for generating sentiment scores
def get_sentiment_scores(news_item):
    inputs = tokenizer(news_item, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)



    logits = outputs.logits.detach().numpy()[0]

    positive_logits = logits[0]

    negative_logits = logits[1]

    # Applying softmax to get scores from 0 to 1
    logits_for_pos_neg = np.array([positive_logits, negative_logits])
    exp_logits = np.exp(logits_for_pos_neg - np.max(logits_for_pos_neg))

    probs = exp_logits / np.sum(exp_logits)
    return({"positive": probs[0], "negative": probs[1]})




#fn to  store sentiment scores
def generate_sentiments(grouped_data):
    sentiments_list = []

    for stock_news in grouped_data['News']:
        sentiment_scores = get_sentiment_scores(stock_news)
        sentiments_list.append(sentiment_scores)

    #converting list of dictionaries into a DataFrame and merging with  original data
    sentiment_df = pd.DataFrame(sentiments_list)
    grouped_data = grouped_data.reset_index(drop=True)
    result_df = pd.concat([grouped_data, sentiment_df], axis=1)

    return result_df

data = pd.read_csv('/content/stock_news.csv')


data['Date'] = pd.to_datetime(data['Date'])
data['Week'] = data['Date'].dt.isocalendar().week

# Grouping the data in  weeks
data_grouped = data.groupby('Week')


data = []

for week, grouped_data in data_grouped:
    week_sentiments = generate_sentiments(grouped_data)
    data.append(week_sentiments)


data = pd.concat(data, axis=0, ignore_index=True)

data = data.drop(columns = (data.columns[[0,2,3, 4, 5,6,7]]))


datap = data.groupby('Week', group_keys=False).apply(
    lambda x: x.sort_values('positive', ascending=False).head(3)
).reset_index(drop=True)



datap["Top_3_Positive_Events"]  = datap["News"]
datap["Positive_Score(%)"]= datap["positive"]
datap=datap.drop(columns=["positive"])
datap= datap.drop(columns=["News"])
datap = datap.drop(columns=["negative"])




datan = data.groupby('Week', group_keys=False).apply(
    lambda x: x.sort_values('negative', ascending=False).head(3)
).reset_index(drop=True)


datan["Top_3_Negative_Events"]  = datan["News"]
datan["Negative_Score(%)"]= datan["negative"]

datan = datan.drop(columns=["News"])
datan=datan.drop(columns=["negative"])

datan = datan.drop(columns=["positive"])




datap["Top_3_Negative_Events"] = datan["Top_3_Negative_Events"]
datap["Negative_Score(%)"] = datan["Negative_Score(%)"]


datap.to_csv("stock_sentiment.csv",index = False)

