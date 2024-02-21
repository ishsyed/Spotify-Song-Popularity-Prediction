import pandas as pd
import configparser
import spotipy
import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials
from utils import get_album_list, song_list_features, predict_popularity, show_predicted_table
import warnings
warnings.filterwarnings('ignore')
    
def get_spotipy_client():
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini')
    cid = config.get('SPOTIFY', 'customer_id')
    secret = config.get('SPOTIFY', 'secret_id')
    client_credentials_manager = SpotifyClientCredentials(client_id=cid,
                                                        client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    sp.trace = False
    return sp

def show_predict_page():
    st.title("Songs Popularity Prediction")

    sp = get_spotipy_client()
    artist_list = []
    artist_name_id_df = pd.read_csv('./data/artist_name_id.csv')
    select_dict = {"Select...": None}
    artist_name_id = dict(artist_name_id_df[['artist_name', 'artist_id']].values)
    artist_name_id = {**select_dict, **artist_name_id}
    
    artist_list.extend(list(artist_name_id.keys()))
    artist_name = st.selectbox("Name of the Artist", artist_list)

    album_list = []
    artist_album_id = get_album_list(sp, artist_name_id[artist_name])
    album_list.extend(list(artist_album_id.keys()))
    album_name = st.selectbox("Name of the Album", album_list)
    st.write("")
    btn_click = st.button("Predict")
    st.write("")

    if btn_click:
        print("Fetching features...")
        songs_features = song_list_features(sp, artist_album_id[album_name])
        songs_features.reset_index(inplace=True, drop=True)
        print(f"Songs Features::: \n {songs_features.head(5)}")
        songs_features.to_csv("./data/song_features.csv")
        print("Prediciting...")
        songs_features.rename({"id": "track_id"}, axis=1, inplace=True)
        prediction_df = predict_popularity(songs_features)
        print(f"Output::: \n {prediction_df.head(5)}")
        prediction_df.to_csv('./data/predictions.csv', index=False)
        show_predicted_table(prediction_df)

if __name__ == '__main__':
    st.set_page_config('Popularity Prediction')
    show_predict_page()
