from urlextract import URLExtract
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages, len(links)


#most busy user

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

#wordcloud
def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc




def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df



def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df



def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap






analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(message):
    score = analyzer.polarity_scores(message)
    return score['compound']

def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['sentiment_score'] = df['message'].apply(get_sentiment_score)
    df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

    sentiment_counts = df['sentiment'].value_counts()
    overall_score = df['sentiment_score'].mean()

    summary = {
        "sentiment_counts": sentiment_counts.to_dict(),
        "overall_score": overall_score,
        "overall_sentiment": (
            "Positive 😊" if overall_score > 0.05 else
            "Negative 😞" if overall_score < -0.05 else
            "Neutral 😐"
        )
    }
    return summary, df


def plot_sentiment_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x='sentiment', data=df, palette='coolwarm', ax=ax)
    ax.set_title('Sentiment Distribution')
    return fig



def message_frequency_per_user(df):
    return df['user'].value_counts()

def average_message_length_per_user(df):
    df['message_length'] = df['message'].apply(len)
    return df.groupby('user')['message_length'].mean().sort_values(ascending=False)


extract = URLExtract()

def media_link_senders(df):
    media_counts = df[df['message'] == '<Media omitted>\n'].groupby('user').count()['message']
    link_counts = df['message'].apply(lambda msg: len(extract.find_urls(msg)))
    df['link_count'] = link_counts
    link_counts_by_user = df.groupby('user')['link_count'].sum()

    return media_counts.sort_values(ascending=False), link_counts_by_user.sort_values(ascending=False)

def response_time_analysis(df):
    df = df[df['user'] != 'group_notification']
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate time difference from previous message
    df['prev_user'] = df['user'].shift(1)
    df['prev_time'] = df['date'].shift(1)
    df['response_time'] = df['date'] - df['prev_time']

    # Only consider actual replies (different user)
    df = df[df['user'] != df['prev_user']]

    # Convert timedelta to seconds
    df['response_time_sec'] = df['response_time'].dt.total_seconds()

    # Average response time by user
    avg_response = df.groupby('user')['response_time_sec'].mean().sort_values()

    fastest = avg_response.idxmin()
    slowest = avg_response.idxmax()

    return avg_response, fastest, slowest




#user interaction graph

def build_interaction_graph(df):
    df = df[df['user'] != 'group_notification']
    df = df.sort_values('date').reset_index(drop=True)

    edges = []

    for i in range(1, len(df)):
        sender = df.loc[i, 'user']
        receiver = df.loc[i-1, 'user']
        if sender != receiver:
            edges.append((sender, receiver))

    # Build directed graph with edge weights (number of replies)
    G = nx.DiGraph()
    for edge in edges:
        if G.has_edge(*edge):
            G[edge[0]][edge[1]]['weight'] += 1
        else:
            G.add_edge(edge[0], edge[1], weight=1)

    return G



#centrality measures

def calculate_centrality_measures(G):
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    eigen = nx.eigenvector_centrality(G, max_iter=500)

    df = pd.DataFrame({
        'User': list(degree.keys()),
        'Degree Centrality': list(degree.values()),
        'Betweenness Centrality': list(betweenness.values()),
        'Eigenvector Centrality': list(eigen.values())
    })

    # Rank by eigenvector centrality
    most_central_user = df.loc[df['Eigenvector Centrality'].idxmax(), 'User']
    return df.sort_values('Eigenvector Centrality', ascending=False), most_central_user


def draw_interaction_graph(G):

    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray',
            width=weights, arrows=True, arrowstyle='->', arrowsize=15)
    plt.title("User Interaction Graph (Replies)")
    return plt

