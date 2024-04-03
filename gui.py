import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image


def Gui(output=pd.read_csv("output_file.csv")):
  st.set_page_config(page_title='Stock Market Analysis')
  st.header('Stock Market Sentiment Analysis')
  st.subheader('Analysis of the sentiment of the stocks this week')
  st.dataframe(output)
  # piechart = px.pie(output,
  #                   values='content_polarity',
  #                   names='stock_names',
  #                   title='Pie Chart')
  # st.plotly_chart(piechart)
  fig = px.bar(output,
               x='stock_names',
               y='content_polarity',
               color='content_polarity',
               color_continuous_scale='RdBu',
               labels={'content_polarity': 'content_polarity'},
               title='Sentiment Analysis of Stocks')
  fig.update_layout(yaxis_range=[-1, 1],
                    coloraxis_colorbar=dict(title='Polarity Score'))
  st.plotly_chart(fig)
  