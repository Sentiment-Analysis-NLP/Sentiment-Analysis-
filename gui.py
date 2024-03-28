import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image


def Gui(output=pd.read_csv("output_file.csv")):
  st.set_page_config(page_title='Stock Market Analysis')
  st.header('Stock Market Sentiment Analysis')
  st.subheader('Analysis of the sentiment of the stocks this week')
