# Libraries
import pandas as pd                 
import io                            
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st 

with open('movies_data.csv','rb') as f:
    data = pd.read_csv(io.BytesIO(f.read()), encoding='unicode_escape')

# printing head of datapip 
print("Original Head of the Data : \n")
print(data.head())
print("\n ----- \n")

# dropped the null values from data
data.dropna(inplace=True)

# droping duplicated values
data.drop_duplicates(inplace=True)

# For year
data['Year'] = data['Year'].str.replace(r'[()]', '', regex=True).astype(int)

# For Duration
data['Duration'] = pd.to_numeric(data['Duration'].str.replace(' min', ''))

# For genre 
data['Genre'] = data['Genre'].str.split(', ')
data = data.explode('Genre')
data['Genre'].fillna(data['Genre'].mode()[0], inplace=True)

# For Vote : convert value into numeric values
data['Votes'] = pd.to_numeric(data['Votes'].str.replace(',', ''))

print(data.head())

st.title(":green[Task 2 : Movie Rating Prediction]")

# Visualization of data : 
t1,t2,t3= st.tabs(["Visualization","Prediction","Performance"])

with t1:
    tabs = st.tabs(["Year","AVG_Rating","Distribution_of_Rating"])
    with tabs[0]:
# Histogram
        st.subheader("Histogram")
        year = px.histogram(data, x ='Year',histnorm='probability density',nbins= 15)
        year.update_traces(marker_color='#8ab8e0')
        year.update_layout(width=800,height=400)
        st.plotly_chart(year)

    with tabs[1]:
        # Line chart
        # First grouping year and genre for rating
        rate_avg_year = data.groupby(['Year','Genre'])['Rating'].mean().reset_index()

        # First ten data
        first_ten_data = data['Genre'].value_counts().head(10).index

        # filtering 
        filtered_data = rate_avg_year[rate_avg_year['Genre'].isin(first_ten_data)]

        # line chart
        faceted_line_chart = px.line(rate_avg_year, x='Year', y='Rating', color='Genre', facet_col='Genre', facet_col_wrap=3,
                                     title='Faceted Line Chart of Average Rating by Year for Genre')
        faceted_line_chart.update_layout(width=900,height=600, xaxis_title='Year', yaxis_title='Average Rating')
        st.plotly_chart(faceted_line_chart)                                                     

    with tabs[2]:

        # distribution

        rating = px.histogram(data, x='Rating', histnorm='probability density', nbins=40)
        rating.update_traces(marker_color='#8ab8e0')
        rating.update_layout(title="Distribution of Rating", title_x=0.5)
        st.plotly_chart(rating)


data.drop('Name',axis=1,inplace=True)

#grouping columns

mean_of_genre_rating = data.groupby('Genre')['Rating'].transform('mean')
data['Mean_of_genre_rating'] = mean_of_genre_rating

mean_of_director_rating = data.groupby('Director')['Rating'].transform('mean')
data['Mean_of_director_rating'] = mean_of_director_rating

mean_of_actor1_rating = data.groupby('Actor 1')['Rating'].transform('mean')
data['Mean_of_actor1_rating'] = mean_of_actor1_rating

mean_of_actor2_rating = data.groupby('Actor 2')['Rating'].transform('mean')
data['Mean_of_actor2_rating'] = mean_of_actor2_rating

mean_of_actor3_rating = data.groupby('Actor 3')['Rating'].transform('mean')
data['Mean_of_actor3_rating'] = mean_of_actor3_rating

with t2:

    # For model
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score

    # making our input and output variables

    input_x = data[['Year','Votes','Duration','Mean_of_genre_rating','Mean_of_director_rating','Mean_of_actor1_rating','Mean_of_actor2_rating','Mean_of_actor3_rating']]
    output_y = data['Rating']

    # train and test data
    x_train,x_test,y_train,y_test = train_test_split(input_x,output_y,test_size=0.2,random_state=42)

    # building the model
    model = LinearRegression()

    # fitting data
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    # testing

    input_year = (st.number_input("Enter Year Value"))
    input_votes = (st.number_input("Enter Vote Value"))
    input_duration = (st.number_input("Enter Duration Value "))
    input_mean_genre_rating = (st.number_input("Enter Genre Mean Rating"))
    input_director = (st.number_input("Enter Director encoded value "))
    input_actor1 = (st.number_input("Enter Actor 1 encoded value"))
    input_actor2 = (st.number_input("Enter Actor 2 encoded value"))
    input_actor3 = (st.number_input("Enter Actor 3 encoded value"))

    test = pd.DataFrame({'Year':[input_year],'Votes':[input_votes],'Duration':[input_duration],
                        'Mean_of_genre_rating':[input_mean_genre_rating],
                        'Mean_of_director_rating':[input_director],
                        'Mean_of_actor1_rating':[input_actor1],
                        'Mean_of_actor2_rating':[input_actor2],
                        'Mean_of_actor3_rating':[input_actor3]})


    if st.button("Make Prediction For Rating"):

        prediction = model.predict(test)
        st.subheader("Rate of The Movie is ")
        st.write(prediction[0])

with t3:

    # Performance of the model
    y_pred = model.predict(x_test)

    st.subheader('The Performance Evaluation with the Logistic Linear Regression is below')

    st.write('Mean Square Error : ',mean_squared_error(y_test,y_pred)*100)
    st.write('Mean Absolute Error : ',mean_absolute_error(y_test,y_pred)*100)
    st.write('R2 score : ',r2_score(y_test,y_pred)*100)