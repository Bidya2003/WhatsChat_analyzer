import streamlit as st
import matplotlib.pyplot as plt
import preprocessor , helper, ner
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Whatsapp Chat Analyzer", layout="wide")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)


    #fetch uniq users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show Analysis WRT" , user_list)

    if st.sidebar.button("show Analysis"):

        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        st.title("Basic Statistical Features")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)


        st.title("Time-Based Features")

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

         # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)



        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)



        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)



        # finding the busiest users in the group(Group level)

        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)


        st.title("Text Analysis")
       

        #wordcloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)


        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)






        # sentiment analysis
        st.title("Sentiment Analysis")

        sentiment_summary, sentiment_df = helper.sentiment_analysis(selected_user, df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Positive Messages", sentiment_summary['sentiment_counts'].get('positive', 0))
        with col2:
            st.metric("Negative Messages", sentiment_summary['sentiment_counts'].get('negative', 0))
        with col3:
            st.metric("Neutral Messages", sentiment_summary['sentiment_counts'].get('neutral', 0))

        st.write(f"**Overall Sentiment:** {sentiment_summary['overall_sentiment']}")
        st.write(f"**Average Sentiment Score:** {sentiment_summary['overall_score']:.2f}")

        # Sentiment distribution plot
        fig = helper.plot_sentiment_distribution(sentiment_df)
        st.pyplot(fig)




        # USER BEHAVIOR ANALYSIS
        st.title("User Behaviit statusor Analysis")

        # 1. Message frequency
        st.subheader("1. Message Frequency Per User")
        msg_freq = helper.message_frequency_per_user(df)
        st.bar_chart(msg_freq)

        # 2. Average message length
        st.subheader("2. Average Message Length Per User")
        avg_len = helper.average_message_length_per_user(df)
        st.bar_chart(avg_len)

        # 3. Media/Link Senders
        st.subheader("3. Media and Link Senders")
        media_counts, link_counts = helper.media_link_senders(df)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Media Messages**")
            st.dataframe(media_counts)
        with col2:
            st.write("**Links Shared**")
            st.dataframe(link_counts)

        # 4. Response Time
        st.subheader("4. Response Time Analysis")
        response_df, fastest, slowest = helper.response_time_analysis(df)
        st.dataframe(response_df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fastest Responder", fastest)
        with col2:
            st.metric("Slowest Responder", slowest)




        #network analysis

        st.title("Network Analysis")

        # Build Graph
        G = helper.build_interaction_graph(df)

        # Centrality
        centrality_df, most_central = helper.calculate_centrality_measures(G)
        st.subheader("Centrality Measures")
        st.dataframe(centrality_df)

        st.success(f"💡 Most Central User: **{most_central}**")

        # Draw graph
        st.subheader("User Interaction Graph")
        fig = helper.draw_interaction_graph(G)
        st.pyplot(fig)





        
        # Named Entity Recognition (NER) Section
        # ---------------------------------------------
        st.title("📌 Named Entity Recognition (NER)")

        # Extract relevant messages based on user selection
        messages = df[df['user'] == selected_user]['message'].tolist() if selected_user != "Overall" else df['message'].tolist()

        # Run NER
        entities = ner.extract_named_entities(messages)

        # Most Mentioned People
        st.subheader("Most Mentioned People")
        top_people = ner.get_top_entities(entities, "PERSON")
        if top_people:
            people_df = pd.DataFrame(top_people, columns=["Name", "Mentions"])
            st.dataframe(people_df)
        else:
            st.info("No person names detected in the messages.")

        # Most Mentioned Locations
        st.subheader("Most Mentioned Locations")
        top_places = ner.get_top_entities(entities, "GPE")
        if top_places:
            places_df = pd.DataFrame(top_places, columns=["Location", "Mentions"])
            st.dataframe(places_df)
        else:
            st.info("No geographic locations detected in the messages.")









