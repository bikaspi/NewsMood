

```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Search Term
target_terms = ("@BBCNews", "@CBSNews", "@CNN",
                "@FoxNews", "@nytimes")
```


```python
# Define a function to return sentiment for all search terms
def news_sentiment(target):
    counter = 0
    
    # variables for holding sentiment
    sentiments = []
    # Run search around each tweet
    news_tweets = api.user_timeline(target, count=100)
    
    # Loop through all tweets
    for tweet in news_tweets:
        # Run Vader Analysis on each tweet
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = counter
        
        # Add sentiments for each tweet into an array
        sentiments.append({"twitter source account" : target,
            "Tweets ago" : tweets_ago,
            "Date" : tweet['created_at'],
            "Compound" : compound,
            'Negative' : neg,
            "Positive"  : pos,
            "Neutral"  : neu,
            "tweet_text" : tweet["text"]})
        

        # Add to counter 
        counter = counter + 1
        
    # Create a dataframe from sentiments dictionary
    data= pd.DataFrame.from_dict(sentiments)
    data.sort_index(axis=0 ,ascending=True, inplace = True)  
    return  data
        
```


```python
# Extracting BBC news sentiment from the dataframe
bbc = news_sentiment('@BBCNews')

# Use datetime library to convert Data stored in string to datetime format
bbc['Date'] = pd.to_datetime(bbc['Date'])
bbc.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets ago</th>
      <th>tweet_text</th>
      <th>twitter source account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.2960</td>
      <td>2018-04-09 21:24:38</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.0</td>
      <td>0</td>
      <td>Tuesday's Telegraph: "'Act now to stop chemica...</td>
      <td>@BBCNews</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.6597</td>
      <td>2018-04-09 21:19:39</td>
      <td>0.286</td>
      <td>0.714</td>
      <td>0.0</td>
      <td>1</td>
      <td>Tuesday's i: "UK plan to beat prostate cancer"...</td>
      <td>@BBCNews</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.4019</td>
      <td>2018-04-09 21:19:35</td>
      <td>0.213</td>
      <td>0.787</td>
      <td>0.0</td>
      <td>2</td>
      <td>Tuesday's Express: "Revealed: shocking treatme...</td>
      <td>@BBCNews</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>2018-04-09 21:07:56</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>3</td>
      <td>Tuesday's Metro: "Shrine to burglar... outside...</td>
      <td>@BBCNews</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.3182</td>
      <td>2018-04-09 21:07:48</td>
      <td>0.141</td>
      <td>0.859</td>
      <td>0.0</td>
      <td>4</td>
      <td>Tuesday's FT: "Russian stocks battered by new ...</td>
      <td>@BBCNews</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extracting CBS news sentiment from the dataframe
cbs = news_sentiment('@CBSNews')
cbs['Date'] = pd.to_datetime(cbs['Date'])
cbs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets ago</th>
      <th>tweet_text</th>
      <th>twitter source account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.6705</td>
      <td>2018-04-09 21:21:49</td>
      <td>0.000</td>
      <td>0.756</td>
      <td>0.244</td>
      <td>0</td>
      <td>Los Angeles is painting some of its roads whit...</td>
      <td>@CBSNews</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3612</td>
      <td>2018-04-09 20:54:07</td>
      <td>0.000</td>
      <td>0.884</td>
      <td>0.116</td>
      <td>1</td>
      <td>"Whether this is related to the [special couns...</td>
      <td>@CBSNews</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>2018-04-09 20:37:01</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2</td>
      <td>NEW: "The decision by the US Attorney’s Office...</td>
      <td>@CBSNews</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.4448</td>
      <td>2018-04-09 20:35:49</td>
      <td>0.128</td>
      <td>0.872</td>
      <td>0.000</td>
      <td>3</td>
      <td>MORE: "It is unclear whether this is stemming ...</td>
      <td>@CBSNews</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>2018-04-09 20:32:27</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>4</td>
      <td>"We did learn that the seizure by FBI and law ...</td>
      <td>@CBSNews</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extracting CNN news sentiment from the dataframe
cnn = news_sentiment('@CNN')
cnn['Date'] = pd.to_datetime(cnn['Date'])
cnn.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets ago</th>
      <th>tweet_text</th>
      <th>twitter source account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>2018-04-09 21:25:46</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0</td>
      <td>The nonpartisan Congressional Budget Office ha...</td>
      <td>@CNN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.4588</td>
      <td>2018-04-09 21:12:14</td>
      <td>0.218</td>
      <td>0.704</td>
      <td>0.077</td>
      <td>1</td>
      <td>The biggest Black Lives Matter page on Faceboo...</td>
      <td>@CNN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>2018-04-09 21:00:54</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2</td>
      <td>First Fortune 500 Latina CEO Geisha Williams t...</td>
      <td>@CNN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.4215</td>
      <td>2018-04-09 20:54:26</td>
      <td>0.209</td>
      <td>0.697</td>
      <td>0.094</td>
      <td>3</td>
      <td>The monster responsible for these attacks has ...</td>
      <td>@CNN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>2018-04-09 20:48:09</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>4</td>
      <td>JUST IN: President Trump has been watching TV ...</td>
      <td>@CNN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extracting FOX news sentiment from the dataframe
fox = news_sentiment('@FoxNews')
fox['Date'] = pd.to_datetime(fox['Date'])
fox.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets ago</th>
      <th>tweet_text</th>
      <th>twitter source account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>2018-04-09 21:28:49</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0</td>
      <td>FBI raids home, office of Trump attorney Micha...</td>
      <td>@FoxNews</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3182</td>
      <td>2018-04-09 21:25:15</td>
      <td>0.122</td>
      <td>0.667</td>
      <td>0.212</td>
      <td>1</td>
      <td>.@kimguilfoyle on alleged chemical attack in S...</td>
      <td>@FoxNews</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>2018-04-09 21:14:17</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2</td>
      <td>UPDATE: FBI raids home, office of #Trump attor...</td>
      <td>@FoxNews</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>2018-04-09 20:51:00</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>3</td>
      <td>Fleetwood Mac replacing Lindsey Buckingham bef...</td>
      <td>@FoxNews</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.8020</td>
      <td>2018-04-09 20:41:00</td>
      <td>0.474</td>
      <td>0.526</td>
      <td>0.000</td>
      <td>4</td>
      <td>Opioid crisis worse than Pablo Escobar era: ex...</td>
      <td>@FoxNews</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extracting NYT news sentiment from the dataframe
nyt = news_sentiment('@nytimes')
nyt['Date'] = pd.to_datetime(nyt['Date'])
nyt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets ago</th>
      <th>tweet_text</th>
      <th>twitter source account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.1027</td>
      <td>2018-04-09 21:32:04</td>
      <td>0.235</td>
      <td>0.542</td>
      <td>0.223</td>
      <td>0</td>
      <td>A Brooklyn man's mind began to unravel after t...</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.2732</td>
      <td>2018-04-09 21:17:06</td>
      <td>0.174</td>
      <td>0.618</td>
      <td>0.208</td>
      <td>1</td>
      <td>RT @ABarnardNYT: New court papers contend top ...</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>2018-04-09 21:00:07</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2</td>
      <td>🤫\nhttps://t.co/ivXQVRb3TB</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5106</td>
      <td>2018-04-09 20:50:07</td>
      <td>0.000</td>
      <td>0.875</td>
      <td>0.125</td>
      <td>3</td>
      <td>RT @MarkLandler: “A pleasant guy can still end...</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>2018-04-09 20:40:12</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>4</td>
      <td>One NYT reader's reaction to President Trump s...</td>
      <td>@nytimes</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Concatenate multiple data frames into a single frame
frames = [bbc, cbs, cnn, fox, nyt]
result = pd.concat(frames)
result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets ago</th>
      <th>tweet_text</th>
      <th>twitter source account</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.2960</td>
      <td>2018-04-09 21:24:38</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.0</td>
      <td>0</td>
      <td>Tuesday's Telegraph: "'Act now to stop chemica...</td>
      <td>@BBCNews</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.6597</td>
      <td>2018-04-09 21:19:39</td>
      <td>0.286</td>
      <td>0.714</td>
      <td>0.0</td>
      <td>1</td>
      <td>Tuesday's i: "UK plan to beat prostate cancer"...</td>
      <td>@BBCNews</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.4019</td>
      <td>2018-04-09 21:19:35</td>
      <td>0.213</td>
      <td>0.787</td>
      <td>0.0</td>
      <td>2</td>
      <td>Tuesday's Express: "Revealed: shocking treatme...</td>
      <td>@BBCNews</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>2018-04-09 21:07:56</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>3</td>
      <td>Tuesday's Metro: "Shrine to burglar... outside...</td>
      <td>@BBCNews</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.3182</td>
      <td>2018-04-09 21:07:48</td>
      <td>0.141</td>
      <td>0.859</td>
      <td>0.0</td>
      <td>4</td>
      <td>Tuesday's FT: "Russian stocks battered by new ...</td>
      <td>@BBCNews</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Export the new CSV as "NewsMedia.csv"
output_fn = "NewsMedia.csv"
result.to_csv(output_fn)
```


```python
# Plotting a scatter plot to visualize polarity analysis of 500 tweets from five different news outlets
plt.figure(figsize=(10,10))
plt.scatter(bbc['Tweets ago'], bbc['Compound'], c='r', alpha = 0.5,  s = 250,  label = 'BBC')
plt.scatter(cbs['Tweets ago'], cbs['Compound'], c='b', alpha = 0.5,  s = 250,  label = 'CBS')
plt.scatter(cnn['Tweets ago'], cnn['Compound'], c='g', alpha = 0.5,  s = 250,  label = 'CNN')
plt.scatter(fox['Tweets ago'], fox['Compound'], c='k', alpha = 0.5,  s = 250,  label = 'FOX')
plt.scatter(nyt['Tweets ago'], nyt['Compound'], c='y', alpha = 0.5,  s = 250,  label = 'NYT')
plt.xlim(105,-0.05)
plt.legend(loc='best')
plt.title('Sentimental Analysis of news media outlets')
plt.xlabel('Tweets Ago')
plt.ylabel('Tweets Polarity')
plt.savefig('PolarityOutput.png')
plt.show()

```


![png](output_10_0.png)



```python
# Calculating mean compound for all 5 news media
avg_compound = {
    'BBC':np.mean(bbc['Compound']),
    'CBS':np.mean(cbs['Compound']),
    'CNN':np.mean(cnn['Compound']),
    'FOX':np.mean(fox['Compound']),
    'NYT':np.mean(nyt['Compound'])
}
avg_compound
```




    {'BBC': -0.051695000000000005,
     'CBS': -0.136475,
     'CNN': -0.154605,
     'FOX': -0.08978100000000001,
     'NYT': -0.013751999999999978}




```python
# compound dataframe
compound_df = pd.DataFrame([avg_compound]).round(3)
compound_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BBC</th>
      <th>CBS</th>
      <th>CNN</th>
      <th>FOX</th>
      <th>NYT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.052</td>
      <td>-0.136</td>
      <td>-0.155</td>
      <td>-0.09</td>
      <td>-0.014</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extracting x_values from the compound dataframe
x_values = list(compound_df)
x_values
```




    ['BBC', 'CBS', 'CNN', 'FOX', 'NYT']




```python
# Extracting y_values from the compound dataframe
y_values = []
for y in x_values:
    y_val = compound_df[y][0]
    y_values.append(y_val)
print(y_values)    
```

    [-0.052, -0.136, -0.155, -0.09, -0.014]



```python
# Plotting a bar chart to represent mean compound 
plt.figure(figsize=(7,5))
plt.bar(x_values,y_values, color = ['r','b','g','k','y'])
plt.title('Bar plot of mean compound polarity')
plt.xlabel('News Media')
plt.ylabel('Tweets polarity')
plt.savefig('BarplotOutput.png')
plt.show()
```


![png](output_15_0.png)



```python
Three observable trends
BBC news seem to send out both postive and negative tweets at the same ratio.
New York Times is mostly positively bias.
The bar plot showed negative polarity for all news media with CNN being the most negatively biased.

```
