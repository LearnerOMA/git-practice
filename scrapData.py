# Here we are using Inida times website to scrap data

from bs4 import BeautifulSoup as bs
import requests as rs
import os
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser;
from sumy.summarizers.lsa import LsaSummarizer;
from sumy.nlp.tokenizers import Tokenizer;
from pprint import pprint
import nltk;

def scrapdata(scrapTopic = "latest"):                                                  
    scrapTopic = scrapTopic.lower().strip().replace(" ", "-");
    url = f"https://timesofindia.indiatimes.com/topic/{scrapTopic}/news"
    response = rs.get(url)
    dataset = []
    if response.status_code == 200:
        soup = bs(response.content, 'html.parser')
        
        
        # Find all articles using the correct class
        articles = soup.find_all(class_='uwU81') 
        
        for article in articles[:20]:           
            # Extract link and headline safely
            article_link = article.a.get('href') if article.a else "No link"
            if article_link != "No link":
                article_link = article_link
            
            headline = article.span.text if article.span else "No headline"
            
            # Extract date from div class 'ZxBIG'
            date_div = article.find('div', class_='ZxBIG')
            if date_div:
                date = date_div.text.strip().split('/')
                date = date[0] if len(date) == 1 else date[1]
            else:
                date = "No date"
            
            # Get the full article content
            full_article = get_full_article(article_link) if article_link != "No link" else "No link"
            summary = Summerize(full_article)  # Mock summary for this example (use summarization tool here)
            
            # Append the article details to the dataset
            dataset.append({
                "Topic": scrapTopic, 
                "Headline": headline.title(),
                "Link": article_link,
                "Date": date,
                "Full_Article": full_article,
                "Article_Summary": summary
            })
        convert_csv_and_dataframe(dataset)
    return dataset



def get_full_article(link):
    new_request = rs.get(link)
    
    if new_request.status_code == 200:
        new_soup = bs(new_request.content, 'html.parser')
        article_body = new_soup.find('div', class_='_s30J clearfix')
        return article_body.text.strip() if article_body else "No article content found."
    else:
        return "Failed to retrieve full article."
    
def convert_csv_and_dataframe(dataset):
    dataframe = pd.DataFrame.from_dict(dataset)
    file_exist = os.path.isfile('dataset.csv')
    dataframe.to_csv('dataset.csv', mode='a', header=not file_exist, index=False)  # Save the dataset as CSV without index
    print(dataframe)  # Display the DataFrame for verification

def Summerize(input_text):
    parser = PlaintextParser.from_string(input_text,Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document,2)
    summarry_text = " ".join([str(sentence) for sentence in summary])       
    return summarry_text



# print("hello")