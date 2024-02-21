import pickle as pkl
import pandas as pd
import streamlit as st

def get_album_list(sp, artist_id):
    if not artist_id:
        return {"Select...": None}
    else:
        album_uri_list = []
        select_dict = {"Select...": None}
        results = sp.artist_albums(artist_id, album_type='album')
        albums = results['items']
        while results['next']:
            results = sp.next(results)
            albums.extend(results['items'])

        for album in albums:
            album_uri_list.append(album['uri'][14:])
        
        album_id_name = pd.DataFrame({'album_id': album_uri_list})
        album_name = []
        for id in album_id_name['album_id']:
            name = sp.album(id)['name']
            album_name.append(name)
        
        album_id_name['album_name'] = album_name
        album_id_name = album_id_name[['album_name', 'album_id']]
        album_id_name = dict(album_id_name[['album_name', 'album_id']].values)
        album_id_name = {**select_dict, **album_id_name}
        return album_id_name

def song_list_features(sp, album_id):
    if not album_id:
        return pd.DataFrame()
    else:
        album_info = sp.album(album_id)
        track_list = []
        track_name = []
        popularity = []
        song = []
        song_popularity = []
        release_date = []
        for i, j in enumerate(album_info['tracks']['items']):
            track_list.append(album_info['tracks']['items'][i]['uri'][14:])
            track_name.append(album_info['tracks']['items'][i]['name'])
            popularity.append(sp.track(album_info['tracks']['items'][i]['uri'][14:])['popularity'])

        song_name_id = pd.DataFrame({'track_name': track_name, 'track_id': track_list, 'popularity': popularity})
        features = pd.DataFrame()
        for id in song_name_id['track_id']:
            features= features.append(pd.DataFrame(sp.audio_features(id)))
        for id in song_name_id['track_name']:
            song.append(id)
        features['track_name'] = song
        for pop in song_name_id['popularity']:
            song_popularity.append(pop)
        features['popularity'] = song_popularity
        for id in song_name_id['track_id']:
            release_date.append(sp.track(id)['album']['release_date'])
        features['release_date'] = release_date
        features['year'] = features['release_date'].apply(lambda x: int(x.split('-')[0]))
        features.drop(['type', 'uri', 'track_href', 'analysis_url', 'release_date'],axis = 1, inplace = True)
        return features
    
def fix_tempo(df):
        df.loc[df['tempo'] == 0 , 'tempo'] = df.loc[df['tempo'] > 0 , 'tempo'].mean()

def convert_popularity(popularity_val):
        return int(popularity_val >= 50)

def classify(value):
    if value == 1:
        return 'Popular'
    else:
        return 'Unpopular'
     
def predict_popularity(df):
    output_df = df[['track_name','popularity']]
    output_df['actual_popularity_category'] = output_df['popularity'].apply(convert_popularity)
    output_df['actual_popularity_category'] = output_df['actual_popularity_category'].apply(classify)

    # We will get rid of the unnecessary columns from the dataset such as track name and track id
    df.drop(['track_name', 'track_id'], axis = 1, inplace = True)
    # We have seen that the tempo needs to be transformed as there are lot of values which are 0 hence transforming them to mean value.
    fix_tempo(df)
    # We also observed that the duration is in milliseconds and hence converting it into minutes
    df['duration_min'] = round(df['duration_ms']/60000, 2)
    df['duration_min'].describe()
    df.drop('duration_ms', axis= 1, inplace= True)
    
    df['popularity'] = df['popularity'].apply(convert_popularity)
    # We are going to encode the year in the dataframe
    with open('./pickle_files/year_encoder_le.pkl', 'rb') as file:
        yr_encoder = pkl.load(file)
    with open('./pickle_files/catcols_encoder_ohe.pkl', 'rb') as file:
        ohe_encoder = pkl.load(file)
    with open('./pickle_files/feature_scaler.pkl', 'rb') as file:
        feature_scaler = pkl.load(file)
    numeric_cols = [ 'danceability', 'energy', 
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_min']
    categorical_cols = ['year','time_signature', 'key']
    df['year'] = yr_encoder.transform(df['year'])
    encoded_new_data_df = pd.DataFrame(ohe_encoder.transform(df[categorical_cols]).toarray(), index=df.index)
    encoded_new_data_df.columns = ohe_encoder.get_feature_names_out()
    new_df = pd.concat([df, encoded_new_data_df], axis=1)
    new_df.drop(categorical_cols, axis=1, inplace=True)
    X = new_df.drop('popularity', axis=1).copy()
    y = new_df['popularity'].copy()
    X[numeric_cols] = feature_scaler.transform(X[numeric_cols])
    with open('./pickle_files/best_catboost_classifier_pkl', 'rb') as cb_file:
        cb_saved = pkl.load(cb_file)
        pred_cb_value = cb_saved.predict(X)
    output_df['predicted_popularity_category'] = list(pred_cb_value)
    output_df['predicted_popularity_category'] = output_df['predicted_popularity_category'].apply(classify)
    return output_df

def show_predicted_table(data):
    st.title("Prediction Analysis")
    st.write("Comparing Actual and Predicted Popularity Category")
    data.rename(columns={"track_name": "Track Name", "popularity": "Popularity", "actual_popularity_category": "Actual Category", "predicted_popularity_category": "Predicted Category"}, inplace=True)
    st.dataframe(data, use_container_width=True)
    accuracy = ((data['Predicted Category']==data['Actual Category']).mean())*100
    st.write(f"Accuracy on the above test data: **{round(accuracy, 2)} %**")
