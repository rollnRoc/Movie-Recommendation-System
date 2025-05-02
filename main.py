from email.mime import application
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from bs4 import BeautifulSoup
import pickle
import requests
import os

# Check if model and vectorizer files exist
def load_models():
    try:
        # Check if files exist first
        if not os.path.exists('nlp_model.pkl') or not os.path.exists('tranform.pkl'):
            print("WARNING: Model files not found! Reviews sentiment analysis will not work.")
            return None, None
        
        # Load the nlp model and tfidf vectorizer from disk
        clf = pickle.load(open('nlp_model.pkl', 'rb'))
        vectorizer = pickle.load(open('tranform.pkl', 'rb'))
        
        # Test vectorizer to make sure it's fitted
        vectorizer.transform(["test sentence"])
        
        return clf, vectorizer
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("Reviews sentiment analysis will not work.")
        return None, None

# Load models at startup
clf, vectorizer = load_models()

def create_similarity():
    try:
        data = pd.read_csv('main_data.csv')
        # creating a count matrix
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data['comb'])
        # creating a similarity score matrix
        similarity = cosine_similarity(count_matrix)
        return data, similarity
    except Exception as e:
        print(f"Error creating similarity matrix: {str(e)}")
        return None, None

# Global variables to store data and similarity matrix
data = None
similarity = None

def rcmd(m):
    global data, similarity
    
    m = m.lower()
    # Load data and similarity matrix if not already loaded
    if data is None or similarity is None:
        data, similarity = create_similarity()
        if data is None or similarity is None:
            return 'Error loading movie data. Please check if main_data.csv exists.'
    
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1], reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    try:
        data = pd.read_csv('main_data.csv')
        return list(data['movie_title'].str.capitalize())
    except Exception as e:
        print(f"Error loading suggestions: {str(e)}")
        return []

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc) == type('string'):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

@app.route("/recommend", methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    try:
        rec_movies = convert_to_list(rec_movies)
        rec_posters = convert_to_list(rec_posters)
        cast_names = convert_to_list(cast_names)
        cast_chars = convert_to_list(cast_chars)
        cast_profiles = convert_to_list(cast_profiles)
        cast_bdays = convert_to_list(cast_bdays)
        cast_bios = convert_to_list(cast_bios)
        cast_places = convert_to_list(cast_places)
        
        # convert string to list (eg. "[1,2,3]" to [1,2,3])
        cast_ids = cast_ids.split(',')
        cast_ids[0] = cast_ids[0].replace("[","")
        cast_ids[-1] = cast_ids[-1].replace("]","")
        
        # rendering the string to python string
        for i in range(len(cast_bios)):
            cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
        
        # combining multiple lists as a dictionary which can be passed to the html file
        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
        casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
        cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return render_template('error.html', error="Error processing movie data")
        
    # web scraping to get user reviews from IMDB site
    movie_reviews = {}
    try:
        print(f"Calling IMDB API: https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt")
        url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}

        response = requests.get(url, headers=headers)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            soup_result = soup.find_all("div", {"class": "ipc-html-content-inner-div"})
            print(f"Found {len(soup_result)} reviews")

            reviews_list = []  # list of reviews
            reviews_status = []  # list of comments (good or bad)
            
            for reviews in soup_result:
                if reviews.string:
                    reviews_list.append(reviews.string)
                    
                    # Only predict sentiment if models are loaded
                    if clf is not None and vectorizer is not None:
                        try:
                            # passing the review to our model
                            movie_review_list = np.array([reviews.string])
                            movie_vector = vectorizer.transform(movie_review_list)
                            pred = clf.predict(movie_vector)
                            reviews_status.append('Good' if pred[0] else 'Bad')
                        except Exception as e:
                            print(f"Error predicting sentiment: {str(e)}")
                            reviews_status.append('Unknown')
                    else:
                        reviews_status.append('Unknown')

            # combining reviews and comments into a dictionary
            movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
        else:
            print(f"Failed to retrieve reviews: {response.status_code}")
    except Exception as e:
        print(f"Error retrieving reviews: {str(e)}")
        # Continue without reviews if there's an error

    # passing all the data to the html file
    return render_template('recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
        vote_count=vote_count, release_date=release_date, runtime=runtime, status=status, genres=genres,
        movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True) 