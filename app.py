import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sn

sn.set_style("darkgrid")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Restaurant Recommendation Web App: (Using Content Based Filtering Technique )")
st.title("Get To Know Which Restaurants You Must Visit In Kolkata: ")
st.image("img.jpg")
# Our dataset:
st.subheader("Our Dataset")
data = pd.read_csv("For Recommending.csv")
data.rename({"voteCount":"zomato_votes"},axis = 1,inplace=True)
st.write(data)
# Restaurant Names
names = list(data["name"].unique())
st.sidebar.header("Which Restaurant You Have Visited ?")
name = st.sidebar.selectbox("Select The Restaurant: ",names)
cusine = data[data["name"] == name]["cusine"].values[0]
st.sidebar.subheader(f"The available cusines in {name} is {cusine}")
st.sidebar.header("Data Analysis For Recommended Restaurants: ")
feature = st.sidebar.selectbox("What you want to analyse ?",("Ratings","Cost","Zomato Votes"))
plot_type = st.sidebar.selectbox("Select A Plot",("Histogram","KDE","Scatter Plot"))
color = st.sidebar.selectbox("Select a color: ",("red","darkblue","limegreen","orange","violet"))

# Creating a 2 functions:
def get_name(index):
    return data[data["Index"] == index]["name"].values[0]
def get_index(name):
    return data[data["name"] == name]["Index"].values[0]
# Creating count matrix:
cv = CountVectorizer()
count_matrix = cv.fit_transform(data["Combined_features"])
# Creating a similarity matrix:
similar_mat = cosine_similarity(count_matrix)
# Get Index of selected name:
index = get_index(name)
similar = list(enumerate(similar_mat[index]))
similar_res = sorted(similar,key = lambda x:x[1] ,reverse = True)
c = 0
st.header("Top 20 best restaurants in kolkata for you: ")
visit = []
for count in similar_res:
    visit.append(get_name(count[0]))
    c += 1
    if c > 20:
        break
st.title(f"As you have visited {name}:")
st.header("You must visit these 20 restaurants in kolkata:")
visit.pop(0)
k = 1
rating = []
cost = []
votes = []
restaurants = []
for i in visit:
    city = data[data["name"] == i]["City"].values[0]
    st.subheader(f"{k}: {i} ({city})")
    rating.append(data[data["name"] == i]["rating"].values[0])
    cost.append(data[data["name"] == i]["cost"].values[0])
    restaurants.append(data[data["name"] == i]["name"].values[0])
    votes.append(data[data["name"] == i]["zomato_votes"].values[0])
    k += 1

series_rating = pd.Series(rating)
s_c = pd.Series(cost)
s_v = pd.Series(votes)

st.title("Data Visualization: ")
if plot_type == "Histogram":
    st.subheader("UNIVARIATE ANALYSIS:")
    if feature == "Ratings":
        st.header("Histogram of ratings of recommended restaurants: ")
        plt.hist(rating,edgecolor = "black",color = color)
        plt.xlabel("Ratings out of 5 ->", color="blue")
        plt.show()
        st.pyplot()
        st.title(f"{feature} For Each Recommended Restaurants: ")
        plt.barh(restaurants, series_rating, color=color)
        plt.show()
        st.pyplot()
    if feature == "Cost":
        st.header("Histogram of cost of recommended restaurants: ")
        plt.hist(s_c, edgecolor="black", color=color)
        plt.xlabel("Per Head Cost ->",color = "blue")
        plt.show()
        st.pyplot()
        st.title(f"Per Head {feature} For Each Recommended Restaurants: ")
        plt.barh(restaurants, s_c, color=color)
        plt.show()
        st.pyplot()

    if feature == "Zomato Votes":
        st.header("Histogram of zomato votes of recommended restaurants: ")
        plt.hist(s_v, edgecolor="black", color=color)
        plt.xlabel("Zomato Votes ->", color="blue")
        plt.show()
        st.pyplot()
        st.title(f"{feature} For Each Recommended Restaurants: ")
        plt.barh(restaurants, s_v, color=color)
        plt.show()
        st.pyplot()

if plot_type == "KDE":
    st.subheader("UNIVARIATE ANALYSIS:")
    if feature == "Ratings":
        st.header("KDE of ratings of recommended restaurants: ")
        st.write(f"The skewness of rating is {round(series_rating.skew(),2)}")
        series_rating.plot(kind = "kde",color = color)
        plt.show()
        st.pyplot()
        st.title(f"{feature} For Each Recommended Restaurants: ")
        plt.barh(restaurants, series_rating,color = color)
        plt.show()
        st.pyplot()
    if feature == "Cost":
        st.header("KDE of cost of recommended restaurants: ")
        st.write(f"The skewness of cost is {round(series_rating.skew(),2)}")
        s_c.plot(kind = "kde",color = color)
        plt.show()
        st.pyplot()
        st.title(f"Per Head {feature} For Each Recommended Restaurants: ")
        plt.barh(restaurants, s_c,color = color)
        plt.show()
        st.pyplot()

    if feature == "Zomato Votes":
        st.header("KDE of zomato votes of recommended restaurants: ")
        st.write(f"The skewness of zomato votes is {round(series_rating.skew(),2)}")
        s_v.plot(kind = "kde",color = color)
        plt.show()
        st.pyplot()
        st.title(f"{feature} For Each Recommended Restaurants: ")
        plt.barh(restaurants, s_v,color = color)
        plt.show()
        st.pyplot()
if plot_type == "Scatter Plot":
    st.subheader("BIVARIATE ANALYSIS:")
    if feature == "Ratings":
        st.header("How Per Head Cost is Changing WRT Ratings: ")
        plt.scatter(series_rating,s_c,color = color)
        plt.xlabel("Ratings ->", size=20, color="blue")
        plt.ylabel("Per Head Cost ->", size=20, color="blue")
        plt.show()
        st.pyplot()
    if feature == "Zomato Votes":
        st.header("How Per Head Cost is Changing WRT Zomato Votings: ")
        plt.scatter(s_v,s_c,color = color)
        plt.xlabel("Zomato Votes ->",size = 20,color = "blue")
        plt.ylabel("Per Head Cost ->",size = 20,color = "blue")
        plt.show()
        st.pyplot()
    if feature == "Cost":
        st.header("No Scatter Plot To Show")


