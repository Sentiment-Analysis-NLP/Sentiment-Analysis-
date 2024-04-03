import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

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
  fig0 = px.bar(output,
                x='stock_names',
                y='content_polarity',
                color='content_polarity',
                color_continuous_scale='RdBu',
                labels={'content_polarity': 'content_polarity'},
                title='Sentiment Analysis of Stocks')
  fig0.update_layout(yaxis_range=[-1, 1],
                     coloraxis_colorbar=dict(title='Polarity Score'))
  st.plotly_chart(fig0)
  st.subheader('Overall market sentiment this week')

  df = pd.DataFrame()
  df['positive'] = output['content_polarity'].apply(lambda x: x
                                                    if x > 0.05 else None)
  df['negative'] = output['content_polarity'].apply(lambda x: x
                                                    if x <= -0.05 else None)
  df['neutral'] = output['content_polarity'].apply(
      lambda x: x if -0.05 < x <= 0.05 else None)

  fig, ax = plt.subplots()
  fig.set_figheight(5)
  fig.set_figwidth(10)

  counts = [
      len(df['positive'].dropna()),
      len(df['neutral'].dropna()),
      len(df['negative'].dropna())
  ]
  percents = [100 * x / sum(counts) for x in counts]

  y_ax = ('Positive', 'Neutral', 'Negative')
  y_tick = np.arange(len(y_ax))

  ax.barh(range(len(counts)),
          counts,
          align="center",
          color=['Red', 'yellow', 'cyan'])
  ax.set_yticks(y_tick)
  ax.set_yticklabels(y_ax, size=15)
  ax.set_facecolor('xkcd:white')
  plt.xlabel('Polarity of Column Body', size=15)

  for i, y in enumerate(ax.patches):
    label_per = percents[i]
    ax.text(y.get_width() + .09,
            y.get_y() + .3,
            str(round((y.get_width()), 1)),
            fontsize=15)
    ax.text(y.get_width() + .09,
            y.get_y() + .1,
            str(f'{round((label_per), 2)}%'),
            fontsize=15)

  sns.despine()

  st.pyplot(fig)
