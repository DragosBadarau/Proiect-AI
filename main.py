import csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from datetime import datetime
import random
from sklearn_som.som import SOM

# used in clustering : title, author, average, num_pages, ratings, text_review, publication, publisher   ->  8
title = []
authors = []
initial_average = []
initial_isbn = []
initial_isbn13 = []
initial_num_pag = []
initial_ratings = []
initial_text_review = []
initial_publication = []
publisher = []
days_difference_list = []
id = []


def get_minimum_date(minimum_date):
    with open('books.csv', newline='', encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[6] == 'eng':
                fulldate = row[10].split("/")
                month = int(fulldate[0])
                day = int(fulldate[1])
                year = int(fulldate[2])
                if year < minimum_date[2]:
                    minimum_date.clear()
                    new_date = [month, day, year]
                    minimum_date = new_date
                elif year == minimum_date[2]:
                    if month < minimum_date[0]:
                        minimum_date.clear()
                        new_date = [month, day, year]
                        minimum_date = new_date
                    elif month == minimum_date[0]:
                        if day < minimum_date[1]:
                            minimum_date.clear()
                            new_date = [month, day, year]
                            minimum_date = new_date
    return minimum_date


def obtain_days_from_minimum(month, day, year, minimum_date):
    date = str(year) + '/' + str(month) + '/' + str(day)
    month1 = str(minimum_date[0])
    day1 = str(minimum_date[1])
    year1 = str(minimum_date[2])
    date1 = str(year1) + '/' + str(month1) + '/' + str(day1)
    d = datetime.strptime(date, "%Y/%m/%d")
    d1 = datetime.strptime(date1, "%Y/%m/%d")
    delta = d - d1
    return delta.days


minimum_date = get_minimum_date([12, 31, 9999])

with open('books.csv', newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if row[6] == 'eng':
            fulldate = row[10].split("/")
            month = int(fulldate[0])
            day = int(fulldate[1])
            year = int(fulldate[2])
            difference_days = obtain_days_from_minimum(month, day, year, minimum_date)
            days_difference_list.append(difference_days)
            id.append(int(row[0]))
            newTitle = re.sub(r"[^a-zA-Z0-9]", " ", row[1].lower())
            title.append(newTitle.replace(' ', ''))
            newAuthors = re.sub(r"[^a-zA-Z0-9]", " ", row[2].lower())
            authors.append(newAuthors.replace(' ', ''))
            initial_average.append(float(row[3]))
            initial_isbn.append(row[4])
            initial_isbn13.append(row[5])
            initial_num_pag.append(float(row[7]))
            initial_ratings.append(float(row[8]))
            initial_text_review.append(float(row[9]))
            initial_publication.append(row[10])
            newPublishers = re.sub(r"[^a-zA-Z0-9]", " ", row[11].lower())
            publisher.append(newPublishers.replace(' ', ''))


# print(days_difference_list)

# print(initial_average)


# min -> 0
# max -> 100
# pt min < nr < max : nr -> (nr-min)/max * 100

# functions for numbers preprocessing
def Average(lst):
    return sum(lst) / len(lst)


def find_min_max(atribute):
    return min(atribute), max(atribute)


def new_value(nr, minn, maxx):
    return (nr - minn) / maxx * 100


def words_to_words_count(attribute):
    new_attribute = []
    for val in attribute:
        new_attribute.append(attribute.count(val))
    return new_attribute


def construct_new_atribute(atribute):
    min_atrib, max_atrib = find_min_max(atribute)
    new_atribute = []
    for x in atribute:
        new_atribute.append(new_value(x, min_atrib, max_atrib))
    return new_atribute


def construct_new_atributes(initial_average, initial_num_pag, initial_ratings, initial_text_review,
                            days_different_list, authors, title, publisher):
    new_average = construct_new_atribute(initial_average)
    new_num_pag = construct_new_atribute(initial_num_pag)
    new_ratings = construct_new_atribute(initial_ratings)
    new_text_review = construct_new_atribute(initial_text_review)
    new_days = construct_new_atribute(days_different_list)
    new_authors = construct_new_atribute(authors)
    new_title = construct_new_atribute(title)
    new_publisher = construct_new_atribute(publisher)
    return new_average, new_num_pag, new_ratings, new_text_review, new_days, new_authors, new_title, new_publisher


def construct_new_list_of_atributes(*atributes):
    new_list = []
    if len(atributes[0]) < 5000:
        size_of_data = len(atributes[0])
    else:
        size_of_data = 5000
    for i in range(0, size_of_data):
        value = []
        for atrb in atributes:
            value.append(atrb[i])
        new_list.append(value)
    return new_list


# filtering functions
def transform_date_in_days(date):
    fulldate = date.split("/")
    month = int(fulldate[0])
    day = int(fulldate[1])
    year = int(fulldate[2])
    days = obtain_days_from_minimum(month, day, year, minimum_date)
    return days


def filter_attribute(attribute, smaller=False, bigger=False, smaller_than=None, bigger_than=None):
    if bigger_than is str:
        # date
        bigger_than = transform_date_in_days(bigger_than)
        if smaller_than:
            smaller_than = transform_date_in_days(smaller_than)

    new_atribute = []
    if smaller:
        for value in attribute:
            if value < smaller_than:
                new_atribute.append(value)
        atribute = new_atribute
    if bigger:
        for value in attribute:
            if value > bigger_than:
                new_atribute.append(value)

    return new_atribute


def remove_book_id(i):
    id.pop(i)
    initial_average.pop(i)
    initial_ratings.pop(i)
    initial_num_pag.pop(i)
    initial_text_review.pop(i)
    days_difference_list.pop(i)
    authors.pop(i)
    publisher.pop(i)
    title.pop(i)


def filter_all_attributes(attribute, smaller=False, bigger=False, smaller_than=None, bigger_than=None):
    new_atribute = filter_attribute(attribute, smaller, bigger, smaller_than, bigger_than)
    nr_of_pops = 0
    for i, val in enumerate(attribute):
        if val not in new_atribute:
            remove_book_id(i - nr_of_pops)
            nr_of_pops += 1


def more_than_average():
    avg = Average(initial_average)
    filter_all_attributes(initial_average, False, True, None, avg)


# clustering functions


def dbscan(x, id, max_distance=2, min_neighbors=3):
    print("DBSCAN : ")
    X = np.array(x)
    clustering = DBSCAN(eps=max_distance, min_samples=min_neighbors).fit(X)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(set(labels))
    random_label = random.randint(0, max(labels))
    association = list(zip(id, labels))
    books_with_random_label = []
    for book in association:
        if book[1] == random_label:
            books_with_random_label.append(book)
    print(f'Label used : {random_label} {books_with_random_label}')
    print(f'Number of clusters: {n_clusters_}; Number of noise points: {n_noise_}')
    return association


def kmeans(x, id, k):
    print("KMeans: ")
    Kmean = KMeans(n_clusters=k)
    X = np.array(x)
    Kmean.fit(X)
    labels = Kmean.labels_
    centroizi = Kmean.cluster_centers_
    print(f'Centroizi {centroizi} ')
    association = list(zip(id, labels))
    return association


def gmm(x, id, nr_components):
    print("GMM")
    X = np.array(x)
    clustering = GaussianMixture(n_components=nr_components).fit(X)
    labels = clustering.predict(X)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(set(labels))
    print(n_clusters_, n_noise_)
    return list(zip(id, labels))


def clusterizare_ierarhica(x, id, link, nr_clusters):
    print("Hierarchical clusterization")
    X = np.array(x)
    clustering = AgglomerativeClustering(n_clusters=nr_clusters, linkage=link).fit(X)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(set(labels))
    print(n_clusters_, n_noise_)
    return list(zip(id, labels))

def SOMAlgorithm(x,id,nr_clusters):
    # Build a 3x1 SOM (3 clusters)
    som = SOM(m=nr_clusters, n=1, dim=8, random_state=1234)
    # Fit it to the data
    X = np.array(x)
    som.fit(X)
    # Assign each datapoint to its predicted cluster
    labels = som.predict(X)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(set(labels))
    print(n_clusters_, n_noise_)
    return list(zip(id, labels))


# filtrare

filter_all_attributes(initial_average, False, True, None, 4)

authors = words_to_words_count(authors)
title = words_to_words_count(title)
publisher = words_to_words_count(publisher)
# print(authors)

# compunere set de date
new_average, new_num_pag, new_ratings, new_text_reviews, new_days, new_authors, new_titles, new_publisher = construct_new_atributes(
    initial_average,
    initial_num_pag,
    initial_ratings,
    initial_text_review,
    days_difference_list, authors, title, publisher)
# print(new_authors)
new_list = construct_new_list_of_atributes(new_average, new_num_pag, new_ratings, new_text_reviews, new_days,
                                           new_authors, new_titles, new_publisher)
# print(len(new_average), len(new_num_pag), len(new_ratings), len(new_text_reviews), len(new_days), len(new_authors), len(new_titles), len(new_publisher))
# print(new_list)

# clusterizare

# print(new_list)
# print(dbscan(new_list, id, 7, 10))
# print(gmm(new_list, id, 4))
# print(clusterizare_ierarhica(new_list, id, 'average', 8))
# print(kmeans(new_list, id, 2))
print(SOMAlgorithm(new_list,id,3))
