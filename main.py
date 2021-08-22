from pandas._libs.tslibs import Timedelta
import streamlit as st
import datetime
import SessionState
import pickle as pkl
import math
import pandas as pd

bus_routes = st.sidebar.multiselect(
    "Select the bus routes: ",
    ("21G","21H",)
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown("# Bus occupancy forecast")

bus_stops = st.sidebar.multiselect("Bus stops: ", ("S1","S2"))
date_input = st.sidebar.date_input('Date: ')
slider_ph = st.sidebar.empty()
tfilter=(datetime.time(0, 0, 0), datetime.time(23, 59, 0))
tfilter = slider_ph.slider("Timeframe to predict", datetime.time.min, datetime.time.max, tfilter, datetime.timedelta(0, 0, 0, 0, 1))

if st.sidebar.button("Predict"):
    
    try:
        for bus_route in bus_routes:
            for bus_stop in bus_stops:

                st.text("Route: " + bus_route)
                st.text("Stop: " + bus_stop)
                model_folder = bus_route + "-" + bus_stop
                st.text("Loading model...")
                model = pkl.load(open(model_folder + "/model.pkl", "rb"))
                start_date = str(date_input) + " " + datetime.time.strftime(tfilter[0], "%H:%M:%S")
                end_date = str(date_input) + " " + datetime.time.strftime(tfilter[1], "%H:%M:%S")
                st.text("Predicting...")
                print(start_date)
                print(end_date)
                y_hat = model.predict(start=start_date,  end=end_date) 

                st.text("")
                st.text("Total expected occupancy for next hour:")
                st.markdown( "### " + str(math.floor(y_hat[0])))
                st.text("Total expected allocation of buses for next hour:")
                if (math.floor(y_hat[0])>30):
                    st.markdown( "### " + str(math.floor(y_hat[0])//30))
                else: 
                    st.markdown( "### " + str(1))
                st.text("")
                st.text("Forecast:")
                st.text(f"{start_date} to {end_date}")
                df_series = pd.Series(y_hat)
                df_series.index = (df_series.index - pd.Timedelta(hours=6, minutes=30)) 
                print( df_series )
                st.line_chart( y_hat.to_numpy() )

    except Exception as e:
        print(e)
        st.text("Bus route / Stop not found")
        st.text("Some error occured!")



