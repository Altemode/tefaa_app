import os
import plotly.graph_objs as go
import plotly.offline as py
import js2py

from st_aggrid import AgGrid
import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from streamlit_vega_lite import altair_component
import altair as alt
import biosignalsnotebooks as bsnb
import csv
import sympy as sp

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.offline as pyoff
#import plotly.graph_objs as go
#import plotly.tools
from plotly.graph_objs import *

import plotly.offline as py

from ipywidgets import interactive, HBox, VBox
from plotly import graph_objects



st.set_page_config(
     page_title="Tefaa Metrics",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }

 )

st.subheader('School of Physical Education and Sports Science')

def main():
    page = st.sidebar.selectbox("Choose a page", ['Prepare you file', 'Insert Entry', 'Calculate Results', 'Center of pressure'])
    


############## ############## PAGE 1 PREPARE THE FILE ############# ############# ############## ##############



    if page == 'Prepare you file':
        st.sidebar.info("From this section you can prepare your txt file with raw data! ")
        st.sidebar.info("First import the file in the form!")
        st.sidebar.info("Secondly choose your preffered time range area to cut!")
        st.sidebar.info("Third choose your preffered time range area to cut!")
        st.sidebar.info("Finaly you may export the file from the 'Export File' button!")
        
        with st.expander("Show File Form"):
            uploaded_file = st.file_uploader("Choose a file")
        platform_mass = st.number_input("Give the platfrom mass:")
        @st.cache(allow_output_mutation=True)
        def get_data():
            if uploaded_file:
                df_raw_data = pd.read_csv(uploaded_file, sep='\s+', skiprows=10, index_col = None)
                #Define Header columns
                columns_count = len(df_raw_data.axes[1])
                if columns_count == 8:
                    df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8']
                if columns_count == 9:
                    df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9']
                if columns_count == 10:
                    df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9', 'Col_10']
                if columns_count == 11:
                    df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9', 'Col_10','Col_11']
                C = 406.831
                #sr = 1000
                resolution = 16
                # Calculate for A Sensor Mass $ Weight
                Vfs_1 = 2.00016
                df_raw_data['Mass_1'] = df_raw_data['Mass_1'] * C / (Vfs_1 * ( (2**resolution) - 1 ) )
                # Calculate for B Sensor Mass $ Weight
                Vfs_2 = 2.00002
                df_raw_data['Mass_2'] = df_raw_data['Mass_2'] * C / (Vfs_2 * ( (2**resolution) - 1 ) )
                # Calculate for C Sensor Mass $ Weight
                Vfs_3 = 2.00057
                df_raw_data['Mass_3'] = df_raw_data['Mass_3'] * C / (Vfs_3 * ( (2**resolution) - 1 ) )
                # Calculate for D Sensor Mass $ Weight
                Vfs_4 = 2.00024
                df_raw_data['Mass_4'] = df_raw_data['Mass_4'] * C / (Vfs_4 * ( (2**resolution) - 1 ) )
                # Calculate the sum of all sensors Mass $ Weight
                df_raw_data['Mass_Sum'] = (df_raw_data['Mass_1'] + df_raw_data['Mass_2'] + df_raw_data['Mass_3'] + df_raw_data['Mass_4']) - platform_mass
                df_raw_data['Rows_Count'] = df_raw_data.index
                return df_raw_data
        df_raw_data= get_data()
        if st.button('Reload Dataframe with Raw Data'):
            get_data()
        if df_raw_data is not None:
            min_time = int(df_raw_data.index.min())
            max_time = int(df_raw_data.index.max())
            selected_time_range = st.slider('Select the whole time range of the graph, per 100', min_time, max_time, (min_time, max_time), 1)
            df_selected_model = (df_raw_data.Rows_Count.between(selected_time_range[0], selected_time_range[1]) )
            df_prepared = pd.DataFrame(df_raw_data[df_selected_model])
            st.line_chart(df_prepared['Mass_Sum'])
            df_prepared.drop(['Rows_Count'], axis = 1, inplace=True)
            filename = uploaded_file.name
            final_filename = os.path.splitext(filename)[0]
            st.write("The file name of your file is : ", final_filename)
            show_df_prepared = st.checkbox("Display the final dataframe")
            if show_df_prepared:
                st.dataframe(df_prepared)
            st.download_button(
                label="Export File",
                data=df_prepared.to_csv(),
                file_name=final_filename +'.csv',
                mime='text/csv',
            )



############# ############## PAGE 2 INSERT TO DATABASE USER+TRIAL ############## ############ #############################



    elif page == 'Insert Entry':

        #Make the connection with Supabase - Database:
        @st.experimental_singleton
        def init_connection():
            url = st.secrets["supabase_url"]
            key = st.secrets["supabase_key"]
            #client = create_client(url, key)
            return create_client(url, key)
        con = init_connection()

        st.sidebar.info("Hello, lets try to insert a new entry to database!")
        st.sidebar.info("-Give the full name of the person!")
        st.sidebar.info("-Give the email adress of the person!")
        st.sidebar.info("-Give the occupy of the person!")
        st.sidebar.info("-Choose the proper kind of trial!")
        st.sidebar.info("-Choose the file of the trial. Please use only undescrores, not spaces in the file name!")
        st.sidebar.info("-Click on Show All Entries to check the database!")
        st.header("Import Entry to Database!")


        #Create the Form to submit data to database:
        with st.form("Create a new entry"):
            fullname = st.text_input("Fullname")
            email = st.text_input("Email address")
            occupy = st.text_input("Occupy")
            kind_of_trial = st.selectbox("Kind of Trial", ('-','Vertical Jump', 'Squat Jump','ISO' ))
            filepath = st.file_uploader("Choose a file")
            #checkbox_val = st.checkbox("Form checkbox")
            submitted = st.form_submit_button("Submit values")
            if submitted:
                filename = filepath.name
                filepath="https://darbnwsgqztqlimdtugr.supabase.co/storage/v1/object/public/files-of-trials/" + filename
                st.write(type(filename))         
                list = (fullname,email,occupy,kind_of_trial,filename)
                def add_entries_to_main_table(supabase):
                    value = {'fullname': fullname, 'email': email, 'occupy': occupy, 'kind_of_trial': kind_of_trial, 'occupy': occupy, 'filename':filename, "filepath": filepath }
                    data = supabase.table('main_table').insert(value).execute()
                def main():
                    new_entry = add_entries_to_main_table(con)
                main()
                st.success('Thank you! A new entry has been inserted to database!')
                st.write(list)
        #@st.experimental_memo(ttl=200)
        def select_all_from_main_table():
            query=con.table("main_table").select("*").execute()
            return query
        main_table_all = select_all_from_main_table()
        df_all_from_main_table = pd.DataFrame(main_table_all.data)

        #Display The whole table :
        display_all_from_main_table = st.checkbox('Show All Entries')
        if display_all_from_main_table:
            st.write(df_all_from_main_table)
        
        url = st.text_input("Paste the desire url")
        #
        if url:
            storage_options = {'User-Agent': 'Mozilla/5.0'}
            df = pd.read_csv(url, sep='\s+',storage_options=storage_options)
            st.write(df)










############## ############## PAGE 3 CALCULATE RESULTS ############# ############# ############## ########################
    
    elif page == 'Calculate Results':
        @st.experimental_singleton
        def init_connection():
            url = st.secrets["supabase_url"]
            key = st.secrets["supabase_key"]
            #client = create_client(url, key)
            return create_client(url, key)
        con = init_connection()

        url_list=[]
        with st.expander("Show all entries"):
            #uploaded_file = st.file_uploader("Choose a file1")
            @st.experimental_memo(ttl=100)
            def select_all_from_main_table():
                query=con.table("main_table").select("*").execute()
                return query
            main_table_all = select_all_from_main_table()
            df_all_from_main_table = pd.DataFrame(main_table_all.data)
            #Display The whole table with persons:
            display_all_from_main_table = st.checkbox('Show All Entries')
            if display_all_from_main_table:
                st.dataframe(df_all_from_main_table)
                #AgGrid(df_all_from_main_table)

            #url_id = st.number_input("Paste the ID of your link",value=0,step=1)
            with st.form("Paste the ID of your link"):   
                    url_id = st.number_input("Paste the ID of your link",value=0,step=1)
                    submitted = st.form_submit_button("Display Results")
            if submitted:
                def select_filepath_from_specific_id():
                    query=con.table("main_table").select("*").eq("id", url_id).execute()
                    return query
                url_query = select_filepath_from_specific_id()   
                url_list =  url_query.data
                
                if url_list:
                    #df_url = pd.DataFrame(url_query.data)
                    url = url_list[0]['filepath']
                    st.write(url_list[0]['filepath'])
                    
                    
                else:
                    st.write("There is no entry with this id")

        #@st.cache(allow_output_mutation=True)
        def get_data():
            if url:
                
                storage_options = {'User-Agent': 'Mozilla/5.0'}
                df = pd.read_csv(url, storage_options=storage_options)
                # #Define Header columns
                columns_count = len(df.axes[1])
                # if columns_count == 8:
                #     df.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8']
                # if columns_count == 9:
                #     df.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9']
                # if columns_count == 10:
                #     df.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9', 'Col_10']
                # if columns_count == 11:
                #     df.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9', 'Col_10','Col_11']
            
                #Define next columns 
                df['pre_pro_signal_EMG_1'] = 0
                df['pre_pro_signal_EMG_2'] = 0
                df['pre_pro_signal_EMG_3'] = 0
                df['RMS_1'] = 0
                df['RMS_2'] = 0
                df['RMS_3'] = 0
                df['Acceleration'] = " "
                df['Start_Velocity'] = " "
                df['Velocity'] = " "
                df['Rows_Count'] = df.index

                # Calculate the sum of all sensors Mass $ Weight
                #df['Mass_Sum'] = (df['Mass_1'] + df['Mass_2'] + df['Mass_3'] + df['Mass_4'])
                pm = df['Mass_Sum'].mean()


                # Calculate The Column Force
                df['Force'] = df['Mass_Sum'] * 9.81
                # Calculate Acceleration
                df['Acceleration'] = (df['Force'] / pm) - 9.81
                # Calculate Velocity
                df['Start_Velocity'] = df.Acceleration.rolling(window=2,min_periods=1).mean()*0.001
                df['Velocity'] = df.Start_Velocity.rolling(window=999999,min_periods=1).sum()

                low_cutoff = 10 # Hz
                high_cutoff = 450 # Hz
                frequency = 1000
                
                if 'Col_9' in df.columns:
                    # THIS IS ALL FOR EMG TO RMS 1
                    # [Baseline Removal] Convert Raw Data EMG to EMG
                    df['Col_9_to_converted'] = (((df['Col_9']/ 2 ** 16) - 1/2 ) * 3 ) / 1000
                    df['Col_9_to_converted'] = df['Col_9_to_converted'] *1000
                    pre_pro_signal_1 = df['Col_9_to_converted'] - df["Col_9_to_converted"].mean()
                    # Application of the signal to the filter. This is EMG1 after filtering
                    pre_pro_signal_1= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal_1, low_cutoff, high_cutoff, frequency)
                    df['pre_pro_signal_EMG_1'] = pre_pro_signal_1**2
                    #This is RMS per 100
                    df['RMS_1'] = df.pre_pro_signal_EMG_1.rolling(window=100,min_periods=100).mean()**(1/2)

                    
                if 'Col_10' in df.columns:
                    # THIS IS ALL FOR EMG TO RMS 2
                    df['Col_10_to_converted'] = (((df['Col_10']/ 2 ** 16) - 1/2 ) * 3 ) / 1000
                    df['Col_10_to_converted'] = df['Col_10_to_converted'] *1000
                    pre_pro_signal_2 = df['Col_10_to_converted'] - df["Col_10_to_converted"].mean()
                    # Application of the signal to the filter. This is EMG1 after filtering
                    pre_pro_signal_2= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal_2, low_cutoff, high_cutoff, frequency)
                    df['pre_pro_signal_EMG_2'] = pre_pro_signal_2**2
                    #This is RMS per 100
                    df['RMS_2'] = df.pre_pro_signal_EMG_2.rolling(window=100,min_periods=100).mean()**(1/2)

                # THIS IS ALL FOR EMG TO RMS 3
                if 'Col_11' in df.columns:
                    df['Col_11_to_converted'] = (((df['Col_11']/ 2 ** 16) - 1/2 ) * 3 ) / 1000
                    df['Col_11_to_converted'] = df['Col_11_to_converted'] *1000
                    pre_pro_signal_3 = df['Col_11_to_converted'] - df["Col_11_to_converted"].mean()
                    # Application of the signal to the filter. This is EMG1 after filtering
                    pre_pro_signal_3= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal_3, low_cutoff, high_cutoff, frequency)
                    df['pre_pro_signal_EMG_3'] = pre_pro_signal_3**2
                    #This is RMS per 100
                    df['RMS_3'] = df.pre_pro_signal_EMG_3.rolling(window=100,min_periods=100).mean()**(1/2)
                
                return pm, df
        

        ############################################################################################################                
        
        
        #st.write(url_list[0]['kind_of_trial'])
        if url_list:
            if url_list[0]['kind_of_trial'] == "Vertical Jump": # & df_url.loc[0, "kind_of_trial"] == "Vertical Jump":
                pm, df = get_data()
                                
                for i in range (0, len(df.index)):
                    if df.loc[i,'Force'] < 5:
                        take_off_time = i
                        break
               
                for i in range (take_off_time, len(df.index)):
                    if df.loc[i,'Force'] > 15:
                        landing_time = i
                        break
            
                for i in range(0,take_off_time):
                    if df.loc[i,'Force'] < (df['Force'].mean() - 20):
                        start_try_time = i
                        break
          

                #Define The Whole Time Range Of Graph
                min_time = int(df.index.min())
                max_time = int(df.index.max())
                with st.expander("Time Range"):
                    selected_time_range = st.slider('Select the whole time range of the graph, per 100', min_time, max_time, (min_time, max_time), 100)
                df_selected_model = (df.Rows_Count.between(selected_time_range[0], selected_time_range[1]) )
                df = pd.DataFrame(df[df_selected_model])
                #Values Sidebar
                with st.sidebar.expander(("Values"), expanded=True):
                    st.write('Name:', url_list[0]['fullname'])
                    st.write('Type of try:', url_list[0]['kind_of_trial'])
                    st.write('Body mass is:', round(pm,2), 'kg')
                    st.write('Occupy:', url_list[0]['occupy'])
                    #st.write('Platform mass is:', round(platform_mass,2), 'kg')
                    st.write('Take Off Time starts at:', take_off_time, 'ms')
                    #st.write('Step for RMS:', rms_step)
                

                kk = df.loc[start_try_time:take_off_time,'Velocity'].sub(0).abs().idxmin()
                ll = (df.loc[kk:take_off_time,'Force']-df['Force'].mean()).sub(0).abs().idxmin()

                #### CREATE THE MAIN CHART #####
                fig = go.Figure()
                # add x and y values for the 1st scatter
                # plot and name the yaxis as yaxis1 values
                fig.add_trace(go.Scatter(
                    x=df['Rows_Count'],
                    y=df['Force'],
                    name="Force",
                    line=dict(color="#290baf")
                    
                ))
                # add x and y values for the 2nd scatter
                # plot and name the yaxis as yaxis2 values
                fig.add_trace(go.Scatter(
                    x=df['Rows_Count'],
                    y=df['Velocity'],
                    name="Velocity",
                    yaxis="y2",
                    line=dict(color="#aa0022")
                ))
                
                # add x and y values for the 3rd scatter
                # plot and name the yaxis as yaxis3 values
                fig.add_trace(go.Scatter(
                    x=df['Rows_Count'],
                    y=df['RMS_1'],
                    name="RMS_1",
                    yaxis="y3"
                ))
                # add x and y values for the 4th scatter plot
                # and name the yaxis as yaxis4 values
                fig.add_trace(go.Scatter(
                    x=df['Rows_Count'],
                    y=df['RMS_2'],
                    name="RMS_2",
                    yaxis="y4",
                    line=dict(color="#7b2b2a")
                ))
                fig.add_trace(go.Scatter(
                    x=df['Rows_Count'],
                    y=df['RMS_3'],
                    name="RMS_3",
                    yaxis="y5",
                    
                ))
                # Create axis objects
                fig.update_layout(
                    # split the x-axis to fraction of plots in
                    # proportions
                    autosize=False,
                    title_text="5 y-axes scatter plot",
                    width=1420,
                    height=550,
                    title_x=0.3,
                    margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    ),
                    plot_bgcolor="#f9f9f9",
                    paper_bgcolor='#f9f9f9',
                    xaxis=dict(
                        domain=[0.125, 0.92],
                        linecolor="#BCCCDC",
                        showspikes=True, # Show spike line for X-axis
                        # Format spike
                        spikethickness=2,
                        spikedash="dot",
                        spikecolor="#999999",
                        spikemode="toaxis",
                        #showspikes= True,
                        #spikemode= 'toaxis' #// or 'across' or 'marker'      
                    ),
                    # pass the y-axis title, titlefont, color
                    # and tickfont as a dictionary and store
                    # it an variable yaxis
                    yaxis=dict(
                        title="Force",
                        titlefont=dict(
                            color="#0000ff"
                        ),
                        tickfont=dict(
                            color="#0000ff"
                        ),
                        linecolor="#BCCCDC",
                        showspikes=True
                    ),
                    # pass the y-axis 2 title, titlefont, color and
                    # tickfont as a dictionary and store it an
                    # variable yaxis 2
                    yaxis2=dict(
                        title="Velocity",
                        titlefont=dict(
                            color="#FF0000"
                        ),
                        tickfont=dict(
                            color="#FF0000"
                        ),
                        anchor="free",  # specifying x - axis has to be the fixed
                        overlaying="y",  # specifyinfg y - axis has to be separated
                        side="left",  # specifying the side the axis should be present
                        position=0.06,  # specifying the position of the axis
                        showspikes=True
                    ),
                    # pass the y-axis 3 title, titlefont, color and
                    # tickfont as a dictionary and store it an
                    # variable yaxis 3
                    yaxis3=dict(
                        title="RMS_1",
                        titlefont=dict(
                            color="#006400"
                        ),
                        tickfont=dict(
                            color="#006400"
                        ),
                        anchor="x",     # specifying x - axis has to be the fixed
                        overlaying="y",  # specifyinfg y - axis has to be separated
                        side="right" # specifying the side the axis should be present
                        #position=0.85
                    ),
                    
                    # pass the y-axis 4 title, titlefont, color and
                    # tickfont as a dictionary and store it an
                    # variable yaxis 4
                    yaxis4=dict(
                        title="RMS_2",
                        titlefont=dict(
                            color="#7b2b2a"
                        ),
                        tickfont=dict(
                            color="#7b2b2a"
                        ),
                        anchor="free",  # specifying x - axis has to be the fixed
                        overlaying="y",  # specifyinfg y - axis has to be separated
                        side="right",  # specifying the side the axis should be present
                        position=0.98  # specifying the position of the axis
                    ),
                    yaxis5=dict(
                        title="RMS_3",
                        titlefont=dict(
                            color="#ffbb00"
                        ),
                        tickfont=dict(
                            color="#ffbb00"
                        ),
                        anchor="free",  # specifying x - axis has to be the fixed
                        overlaying="y",  # specifyinfg y - axis has to be separated
                        side="left",  # specifying the side the axis should be present
                        position=0.00  # specifying the position of the axis
                    )
                )
                # Update layout of the plot namely title_text, width
                # and place it in the center using title_x parameter
                # as shown
                large_rockwell_template = dict(
                    layout=go.Layout(title_font=dict(family="Rockwell", size=24))
                )
                
                #     #template=large_rockwell_template
                #     # barmode='group',
                #     #hovermode='x',#paper_bgcolor="LightSteelBlue"   
                # )
                fig.update_xaxes(
                    
                    rangeslider_visible=True,
                    # rangeselector=dict(
                    #     buttons=list([
                    #         dict(count=1, label="1m", step="month", stepmode="backward"),
                    #         dict(count=4000, label="6m", step="month", stepmode="backward"),
                    #         dict(count=6000, label="YTD", step="year", stepmode="todate"),
                    #         dict(count=12000, label="1y", step="year", stepmode="backward"),
                    #         dict(step="all")
                    #     ])
                    # )
                )
                # fig.add_annotation(
                #     x=kk,
                #     y=df.loc[kk,'Velocity'],
                #     xref="x",
                #     yref="y",
                #     text="max=5",
                #     showarrow=True,
                #     font=dict(
                #         family="Courier New, monospace",
                #         size=16,
                #         color="#ffffff"
                #         ),
                #     align="center",
                #     arrowhead=2,
                #     arrowsize=1,
                #     arrowwidth=2,
                #     arrowcolor="#636363",
                #     ax=20,
                #     ay=-30,
                #     bordercolor="#c7c7c7",
                #     borderwidth=2,
                #     borderpad=4,
                #     bgcolor="#ff7f0e",
                #     opacity=0.8
                #     )
                
                st.plotly_chart(fig)
                
                with st.form("Calculate the Jump"):
                    st.write("The time where the volicity is closest to zero is:", kk)
                    st.write("The time where the Force is closest to average is:", ll)
                    c1, c2= st.columns(2)
                    with c1:                        
                        jump_starts = st.number_input("Jump starts at:",value=kk,step=1)
                    with c2:
                        jump_ends = st.number_input("Jump ends at:",value=ll,step=1)
                    submitted = st.form_submit_button("Calculate")


                if submitted:
                    #df_brushed = df[(df.index >= user_time_input_min_main_table) & (df.index <= user_time_input_max_main_table)]
                    #Find the IMPULSE GRF
                    df['Impulse_grf'] = df.loc[jump_starts:jump_ends, 'Force'] * (1/1000)
                    impulse_grf = df['Impulse_grf'].sum()
                    #Find the IMPULSE BW
                    impulse_bw_duration = (jump_ends - jump_starts) / 1000
                    impulse_bw = pm * 9.81 * impulse_bw_duration
                    velocity_momentum1 = (impulse_grf - impulse_bw) / pm
                    jump_depending_impluse = (velocity_momentum1 ** 2) / (9.81 * 2)
                    st.write("The Jump is:", round(jump_depending_impluse,4), "The Rsi is:", round(impulse_bw_duration/jump_depending_impluse,4))
                
                
                col1, col2 = st.columns(2)
                r=0  
                
                with st.form("Insert Users"):
                    c1, c2= st.columns(2)
                    with c1:        
                        user_time_input_min_main_table = st.number_input("From Time",value=0,step=1)
                    with c2:
                        user_time_input_max_main_table = st.number_input("Till Time",value=0,step=1)
                    submitted = st.form_submit_button("Submit")
                    
                df_brushed = df[(df.index >= user_time_input_min_main_table) & (df.index < user_time_input_max_main_table)]

                ################# BRUSHED AREA ##########################
                if len(df_brushed):
                    df_brushed = df[(df.index >= user_time_input_min_main_table) & (df.index <= user_time_input_max_main_table)]
                    #Find the IMPULSE GRF
                    df_brushed['Impulse_grf'] = df_brushed['Force'] * (1/1000)
                    impulse_grf = df_brushed['Impulse_grf'].sum()
                    #Find the IMPULSE BW
                    impulse_bw_duration = (user_time_input_max_main_table - user_time_input_min_main_table) / 1000
                    impulse_bw = pm * 9.81 * impulse_bw_duration
                    velocity_momentum1 = (impulse_grf - impulse_bw) / pm
                    jump_depending_impluse = (velocity_momentum1 ** 2) / (9.81 * 2)
                    closest_zero_velocity = df_brushed['Velocity'].sub(0).abs().idxmin()

                    #Find the RFD linear igression
                    l_rfd1=[]
                    l_emg1=[]
                    l_emg2=[]
                    l_emg3=[]
                    b_rfd1=[]
                    b_emg1=[]
                    b_emg2=[]
                    b_emg3=[]
                    headers_list_rfd1=[]
                    headers_list_emg1=[]
                    headers_list_emg2=[]
                    headers_list_emg3=[]
                    rfd_df1=pd.DataFrame()
                    emg_df1=pd.DataFrame()
                    emg_df2=pd.DataFrame()
                    emg_df3=pd.DataFrame()
                    for i in range(int(user_time_input_min_main_table),int(user_time_input_max_main_table),50):  
                        #FIND RFD
                        X = df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'] - df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'].mean()
                        Y = df_brushed.loc[user_time_input_min_main_table:i:1,'Force'] - df_brushed.loc[user_time_input_min_main_table:i:1,'Force'].mean()
                        b_rfd1 = (X*Y).sum() / (X ** 2).sum()
                        headers_list_rfd1.append("RFD-"+str(i))
                        l_rfd1.append(b_rfd1)
                        #FIND R-EMG
                        X = df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'] - df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'].mean()
                        Y1 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_1'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_1'].mean()
                        Y2 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_2'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_2'].mean()
                        Y3 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_3'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_3'].mean()

                        b_emg1 = (X*Y1).sum() / (X ** 2).sum()
                        b_emg2 = (X*Y2).sum() / (X ** 2).sum()
                        b_emg3 = (X*Y3).sum() / (X ** 2).sum()

                        headers_list_emg1.append("EMG_1-"+str(i))
                        headers_list_emg2.append("EMG_2-"+str(i))
                        headers_list_emg3.append("EMG_3-"+str(i))
                        l_emg1.append(b_emg1)
                        l_emg2.append(b_emg2)
                        l_emg3.append(b_emg3)

                    if rfd_df1.empty:
                        rfd_df1 = pd.DataFrame([l_rfd1])
                        cols = len(rfd_df1.axes[1])
                        rfd_df1.columns = [*headers_list_rfd1]
                    else:
                        to_append = l_rfd1
                        rfd_df1_length = len(rfd_df1)
                        rfd_df1.loc[rfd_df1_length] = to_append

                    #Dataframe for EMG1
                    if emg_df1.empty:
                        emg_df1 = pd.DataFrame([l_emg1])
                        cols = len(emg_df1.axes[1])
                        emg_df1.columns = [*headers_list_emg1]
                    else:
                        to_append = emg_df1
                        emg_df1_length = len(emg_df1)
                        emg_df1.loc[emg_df1_length] = to_append
                    
                    #Dataframe for EMG2
                    if emg_df2.empty:
                        emg_df2 = pd.DataFrame([l_emg2])
                        cols = len(emg_df2.axes[1])
                        emg_df2.columns = [*headers_list_emg2]
                    else:
                        to_append = emg_df2
                        emg_df2_length = len(emg_df2)
                        emg_df2.loc[emg_df2_length] = to_append

                    #Dataframe for EMG3
                    if emg_df3.empty:
                        emg_df3 = pd.DataFrame([l_emg3])
                        cols = len(emg_df3.axes[1])
                        emg_df3.columns = [*headers_list_emg3]
                    else:
                        to_append = emg_df3
                        emg_df3_length = len(emg_df3)
                        emg_df3.loc[emg_df3_length] = to_append
                    #Give Specific Results
                    with st.expander('Show Specific Calculations', expanded=True):
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                                st.write('Impulse GRF:', round(impulse_grf,4))
                                st.write('Impulse BW:', round(impulse_bw,4))
                                st.write('Net Impulse:', round(impulse_grf - impulse_bw,4))
                                #st.write('velocity_momentum:', round(velocity_momentum1,2))
                                st.write('Jump (Impluse):', round(jump_depending_impluse,4))
                        with col2:
                                st.write('Force-Mean:', round(df_brushed["Force"].mean(),4))
                                st.write('Force-Min:', round(min(df_brushed['Force']),4))
                                st.write('Force-Max:', round(max(df_brushed['Force']),4))
                        with col3:
                                st.write('RMS_1-Mean:', round(df_brushed["RMS_1"].mean(),4))
                                st.write('RMS_2-Mean:', round(df_brushed['RMS_2'].mean(),4))
                                st.write('RMS_3-Mean:', round(df_brushed['RMS_3'].mean(),4))
                        with col4:
                                st.write('Velocity-Mean:', round(df_brushed["Velocity"].mean(),4))
                                st.write('Velocity-Min:', round(min(df_brushed['Velocity']),4))
                                st.write('Velocity-Max:', round(max(df_brushed['Velocity']),4))
                        with col5:
                                st.write('Acceleration-Mean:', round(df_brushed["Acceleration"].mean(),4))
                                st.write('Acceleration-Min:', round(min(df_brushed['Acceleration']),4))
                                st.write('Acceleration-Max:', round(max(df_brushed['Acceleration']),4))
                    
                    #Display Dataframe in Datatable
                    with st.expander("Show Data Table", expanded=True):
                        selected_filtered_columns = st.multiselect(
                        label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS_1', 'RMS_2','RMS_3'), help='Click to select', options=df_brushed.columns)
                        st.write(df_brushed[selected_filtered_columns])
                        #Button to export results
                        st.download_button(
                            label="Export table",
                            data=df_brushed[selected_filtered_columns].to_csv(),
                            file_name='df.csv',
                            mime='text/csv',
                        )
                #The same method for un-brushed Graph
                    st.write('Export All Metrics')
                    specific_metrics = [""]
                    specific_metrics = {#'Unit': ['results'],
                    
                            'Fullname' : url_list[0]['fullname'],
                            'Occupy' : url_list[0]['occupy'],
                            'Type of try' : url_list[0]['kind_of_trial'],
                            'Body Mass (kg)': [pm],
                            'Jump (Velocity Take Off) (m/s)' : [jump_depending_impluse],
                            'Start trial time (s)' : [start_try_time],
                            'Take Off Time (s)' : [take_off_time],
                            'Landing Time (s)' : [landing_time],
                            'Impulse (GRF) (N/s)' : [impulse_grf],
                            'Impulse (BW) (N/s)' : [impulse_bw],
                            'RMS_1 Mean' : [df_brushed['RMS_1'].mean()],
                            'RMS_2 Mean' : [df_brushed['RMS_2'].mean()],
                            'RMS_3 Mean' : [df_brushed['RMS_3'].mean()],
                            'Force Mean (N)' : [df_brushed['Force'].mean()],
                            'Force Max (N)' : [max(df_brushed['Force'])],
                            'Force Min (N)' : [min(df_brushed['Force'])],
                            'Velocity Mean (m/s)' : [df_brushed['Velocity'].mean()],
                            'Velocity Max (m/s)' : [max(df_brushed['Velocity'])],
                            'Velocity Min (m/s)' : [min(df_brushed['Velocity'])],
                            'Acceleration Mean (m^2/s)' : [df_brushed['Acceleration'].mean()],
                            'Acceleration Max (m^2/s)' : [max(df_brushed['Acceleration'])],
                            'Acceleration Min (m^2/s)' : [min(df_brushed['Acceleration'])],
                            }
                    
                    
                    specific_metrics_df = pd.DataFrame(specific_metrics)
                    #specific_metrics_df = specific_metrics_df.round(decimals = 2)

                    #Combine all dataframes to one , for the final export
                    final_results_df = pd.concat([specific_metrics_df, rfd_df1, emg_df1, emg_df2, emg_df3], axis=1, join='inner')
                    #final_results_df['Body Mass (kg)'] = final_results_df['Body Mass (kg)'].round(decimals = 2)
                    final_results_df =np.round(final_results_df, decimals = 4)
                    
                    st.write(final_results_df)
                    
                    #st.write(specific_metrics)
                    st.download_button(
                        label="Export Final Results",
                        data=final_results_df.to_csv(),
                        file_name='final_results.csv',
                        mime='text/csv',
                            )
                else:
                    slider = alt.binding_range(min=0, max=100, step=1, name='cutoff:')
                    with st.expander("Show Specific Calculations", expanded=True):
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                                st.write('Force-Mean:', round(df["Force"].mean(),4))
                                st.write('Force-Min:', round(min(df['Force']),4))
                                st.write('Force-Max:', round(max(df['Force']),4))
                        with col2:
                                st.write('Mass-Mean:', round(df["Mass_Sum"].mean(),4))
                                st.write('Mass-Min:', round(min(df['Mass_Sum']),4))
                                st.write('Mass-Max:', round(max(df['Mass_Sum']),4))
                        with col3:
                                st.write('Velocity-Mean:', round(df["Velocity"].mean(),4))
                                st.write('Velocity-Min:', round(min(df['Velocity']),4))
                                st.write('Velocity-Max:', round(max(df['Velocity']),4))
                        with col4:
                                st.write('RMS_1-Mean:', round(df["RMS_1"].mean(),4))
                                st.write('RMS_2-Mean:', round(df["RMS_2"].mean(),4))
                                st.write('RMS_3-Mean:', round(df["RMS_3"].mean(),4))
                        with col5:
                                st.write('Acceleration-Mean:', round(df["Acceleration"].mean(),4))
                                st.write('Acceleration-Min:', round(min(df['Acceleration']),4))
                                st.write('Acceleration-Max:', round(max(df['Acceleration']),4))
                    data = {'Unit': ['Force', 'Mass_Sum', 'Velocity', 'Acceleration'],
                                'Mean': [df["Force"].mean(), df["Mass_Sum"].mean(), df["Velocity"].mean(), df["Acceleration"].mean()],
                                'Min': [min(df['Force']), min(df['Mass_Sum']), min(df['Velocity']), min(df['Acceleration'])],
                                'Max': [max(df['Force']), max(df['Mass_Sum']), max(df['Velocity']), max(df['Acceleration'])],
                                #'Max': [max(df_brushed['Force']), max(df_brushed['Mass_Sum']), max(df_brushed['Velocity']), max(df_brushed['Acceleration'])] }
                            }               
                    #Display some Values in Sidebar
                    st.sidebar.write('Time range from', min(df['Rows_Count']), 'to', max(df['Rows_Count']), 'ms')
                    st.sidebar.write('Min Mass_Sum:', min(df['Mass_Sum']))
                    st.sidebar.write('Max Mass_Sum:',  max(df['Mass_Sum']))
                    #Display Dataframe in Datatable
                    with st.expander("Show Data Table", expanded=True):
                        selected_clear_columns = st.multiselect(
                        label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS_1','RMS_2'), help='Click to select', options=df.columns)
                        st.write(df[selected_clear_columns])
                        #Button to export results
                        st.download_button(
                            label="Export table",
                            data=df[selected_clear_columns].to_csv(),
                            file_name='df.csv',
                            mime='text/csv',
                        )


        ################# ######################### ISO ##################### ########################## #################
        if 1>2:
            min_time = int(df.index.min())
            max_time = int(df.index.max())
            with st.expander("Time Range"):
                selected_time_range = st.slider('Select the whole time range of the graph, per 100', min_time, max_time, (min_time, max_time), 100)
            df_selected_model = (df.Rows_Count.between(selected_time_range[0], selected_time_range[1]) )
            df = pd.DataFrame(df[df_selected_model])
            @st.cache(allow_output_mutation=True)
            def altair_histogram():
                brushed = alt.selection_interval(encodings=["x"], name="brushed")
                return (
                    alt.Chart(df).transform_fold(
                        ['Force', 'RMS_1', 'RMS_2', 'RMS_3']
                    ).resolve_scale(y='independent')
                    .mark_line().resolve_scale(y='independent')
                    .encode(alt.X("Rows_Count:Q"), y="value:Q", tooltip=['Rows_Count', 'Force', 'Mass_Sum', 'RMS_1', 'RMS_2', 'RMS_3'], color='key:N').add_selection(
                        brushed
                    )
                ).properties(width=900, height=400)
            event_dict = altair_component(altair_chart=altair_histogram())
            r = event_dict.get("Rows_Count")
            #Number input fields to declare time zone for the Table
            col1, col2 = st.columns(2)
            with col1:
                if r:
                    if isinstance(r[0], float) is True:
                        t = int(r[0])
                        user_time_input_min_main_table = st.number_input("From Time ",value=t)
                    else:
                        user_time_input_min_main_table = st.number_input("From Time ",value=r[0])
                else:
                    user_time_input_min_main_table = st.number_input("From Time. ")

            with col2:
                if r:
                    if isinstance(r[1], float) is True:
                        t1 = int(r[1])
                        user_time_input_max_main_table = st.number_input("From Time ",value=t1)
                        filtered = df[(df.Rows_Count >= r[0]) & (df.Rows_Count < r[1])]
                        df_brushed = pd.DataFrame(filtered)
                    else:
                        user_time_input_max_main_table = st.number_input("From Time ",value=r[1])
                        filtered = df[(df.Rows_Count >= r[0]) & (df.Rows_Count < r[1])]
                        df_brushed = pd.DataFrame(filtered)
                else:
                    user_time_input_max_main_table = st.number_input("Till Time. ")
            #Save the brushed dataframe
            df_brushed = df[(df.index >= user_time_input_min_main_table) & (df.index < user_time_input_max_main_table)]
            if len(df_brushed):
                df_brushed = df[(df.index >= user_time_input_min_main_table) & (df.index <= user_time_input_max_main_table)]
                #Find the IMPULSE GRF
                
                
                #Find the RFD linear igression
                l_rfd1=[]
                l_emg1=[]
                l_emg2=[]
                l_emg3=[]
                b_rfd1=[]
                b_emg1=[]
                b_emg2=[]
                b_emg3=[]
                headers_list_rfd1=[]
                headers_list_emg1=[]
                headers_list_emg2=[]
                headers_list_emg3=[]
                rfd_df1=pd.DataFrame()
                emg_df1=pd.DataFrame()
                emg_df2=pd.DataFrame()
                emg_df3=pd.DataFrame()
                for i in range(int(user_time_input_min_main_table),int(user_time_input_max_main_table),50):  
                    X = df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'] - df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'].mean()
                    Y = df_brushed.loc[user_time_input_min_main_table:i:1,'Force'] - df_brushed.loc[user_time_input_min_main_table:i:1,'Force'].mean()
                    XY= X*Y
                    X2= X ** 2 
                    
                    b_rfd1 = XY.sum() / X2.sum()
                    #st.write(round(b_rfd),4)
                    headers_list_rfd1.append("RFD-"+str(i))
                    l_rfd1.append(b_rfd1)
                    #FOR EMG
                    X = df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'] - df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'].mean()
                    Y1 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_1'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_1'].mean()
                    Y2 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_2'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_2'].mean()
                    Y3 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_3'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signal_EMG_3'].mean()
                    XY1= X * Y1
                    X2 = X ** 2
                    #st.write(XY1)
                    
                    b_emg1 = XY1.sum() / X2.sum()
                    b_emg2 = (X*Y2).sum() / (X ** 2).sum()
                    b_emg3 = (X*Y3).sum() / (X ** 2).sum()

                    headers_list_emg1.append("EMG_1-"+str(i))
                    headers_list_emg2.append("EMG_2-"+str(i))
                    headers_list_emg3.append("EMG_3-"+str(i))
                    l_emg1.append(b_emg1)
                    l_emg2.append(b_emg2)
                    l_emg3.append(b_emg3)

                if rfd_df1.empty:
                    rfd_df1 = pd.DataFrame([l_rfd1])
                    cols = len(rfd_df1.axes[1])
                    rfd_df1.columns = [*headers_list_rfd1]
                else:
                    to_append = l_rfd1
                    rfd_df1_length = len(rfd_df1)
                    rfd_df1.loc[rfd_df1_length] = to_append

                #Dataframe for EMG1
                if emg_df1.empty:
                    emg_df1 = pd.DataFrame([l_emg1])
                    cols = len(emg_df1.axes[1])
                    emg_df1.columns = [*headers_list_emg1]
                else:
                    to_append = emg_df1
                    emg_df1_length = len(emg_df1)
                    emg_df1.loc[emg_df1_length] = to_append
                
                #Dataframe for EMG2
                if emg_df2.empty:
                    emg_df2 = pd.DataFrame([l_emg2])
                    cols = len(emg_df2.axes[1])
                    emg_df2.columns = [*headers_list_emg2]
                else:
                    to_append = emg_df2
                    emg_df2_length = len(emg_df2)
                    emg_df2.loc[emg_df2_length] = to_append

                #Dataframe for EMG3
                if emg_df3.empty:
                    emg_df3 = pd.DataFrame([l_emg3])
                    cols = len(emg_df3.axes[1])
                    emg_df3.columns = [*headers_list_emg3]
                else:
                    to_append = emg_df3
                    emg_df3_length = len(emg_df3)
                    emg_df3.loc[emg_df3_length] = to_append
                #Give Specific Results
                with st.expander('Show Specific Calculations', expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                            st.write('Force-Mean:', round(df_brushed["Force"].mean(),4))
                            st.write('Force-Min:', round(min(df_brushed['Force']),4))
                            st.write('Force-Max:', round(max(df_brushed['Force']),4))

                    with col2:
                            st.write('Mass-Mean:', df["Mass_Sum"].mean())
                            st.write('Mass-Min:', min(df['Mass_Sum']))
                            st.write('Mass-Max:', max(df['Mass_Sum']))

                    with col3:
                            st.write('RMS_1-Mean:', round(df_brushed["RMS_1"].mean(),4))
                            st.write('RMS_2-Mean:', round(df_brushed['RMS_2'].mean(),4))
                            st.write('RMS_3-Mean:', round(df_brushed['RMS_3'].mean(),4))
                    
                
                #Display Dataframe in Datatable
                with st.expander("Show Data Table", expanded=True):
                    selected_filtered_columns = st.multiselect(
                    label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum',  'RMS_1', 'RMS_2'), help='Click to select', options=df_brushed.columns)
                    st.write(df_brushed[selected_filtered_columns])
                    #Button to export results
                    st.download_button(
                        label="Export table",
                        data=df_brushed[selected_filtered_columns].to_csv(),
                        file_name='df.csv',
                        mime='text/csv',
                    )
            #The same method for un-brushed Graph
                st.write('Export All Metrics')
                specific_metrics = [""]
                specific_metrics = {#'Unit': ['results'],
                
                        'Fullname' : [fullname],
                        'Type of try' : [type_of_try],
                        'Body Mass (kg)': [pm],
                        'Platform Mass (kg)': [platform_mass],
                        'RMS_1 Mean' : [df_brushed['RMS_1'].mean()],
                        'RMS_2 Mean' : [df_brushed['RMS_2'].mean()],
                        'RMS_3 Mean' : [df_brushed['RMS_3'].mean()],
                        'Force Mean (N)' : [df_brushed['Force'].mean()],
                        'Force Max (N)' : [max(df_brushed['Force'])],
                        'Force Min (N)' : [min(df_brushed['Force'])],
                        }
                
                
                specific_metrics_df = pd.DataFrame(specific_metrics)
                #specific_metrics_df = specific_metrics_df.round(decimals = 2)

                #Combine all dataframes to one , for the final export
                final_results_df = pd.concat([specific_metrics_df, rfd_df1, emg_df1, emg_df2,emg_df3], axis=1, join='inner')
                #final_results_df['Body Mass (kg)'] = final_results_df['Body Mass (kg)'].round(decimals = 2)
                final_results_df =np.round(final_results_df, decimals = 4)
                
                st.write(final_results_df)
                #st.write(specific_metrics)
                st.download_button(
                    label="Export Final Results",
                    data=final_results_df.to_csv(),
                    file_name='final_results.csv',
                    mime='text/csv',
                        )
            else:
                slider = alt.binding_range(min=0, max=100, step=1, name='cutoff:')
                with st.expander("Show Specific Calculations", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                            st.write('Force-Mean:', df["Force"].mean())
                            st.write('Force-Min:', min(df['Force']))
                            st.write('Force-Max:', max(df['Force']))
                    with col2:
                            st.write('Mass-Mean:', df["Mass_Sum"].mean())
                            st.write('Mass-Min:', min(df['Mass_Sum']))
                            st.write('Mass-Max:', max(df['Mass_Sum']))
                    with col3:
                            st.write('RMS_1-Mean:', df["RMS_1"].mean())
                            st.write('RMS_2-Mean:', df["RMS_2"].mean())
                            st.write('RMS_3-Mean:', df["RMS_3"].mean())
                    
                data = {'Unit': ['Force', 'Mass_Sum', 'Velocity', 'Acceleration'],
                            'Mean': [df["Force"].mean(), df["Mass_Sum"].mean(), df["Velocity"].mean(), df["Acceleration"].mean()],
                            'Min': [min(df['Force']), min(df['Mass_Sum']), min(df['Velocity']), min(df['Acceleration'])],
                            'Max': [max(df['Force']), max(df['Mass_Sum']), max(df['Velocity']), max(df['Acceleration'])],
                            #'Max': [max(df_brushed['Force']), max(df_brushed['Mass_Sum']), max(df_brushed['Velocity']), max(df_brushed['Acceleration'])] }
                        }               
                #Display some Values in Sidebar
                st.sidebar.write('Time range from', min(df['Rows_Count']), 'to', max(df['Rows_Count']), 'ms')
                st.sidebar.write('Force-Min:', round(min(df['Force']),4))
                st.sidebar.write('Force-Max:', round(max(df['Force']),4))
                #Display Dataframe in Datatable
                with st.expander("Show Data Table", expanded=True):
                    selected_clear_columns = st.multiselect(
                    label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS_1','RMS_2', 'RMS_3'), help='Click to select', options=df.columns)
                    st.write(df[selected_clear_columns])
                    #Button to export results
                    st.download_button(
                        label="Export table",
                        data=df[selected_clear_columns].to_csv(),
                        file_name='df.csv',
                        mime='text/csv',
                    )
            
            

################# ################ CENTER OF PRESSURE ################ ################## ################# ####################################


    else:
        with st.expander("Show File Form"):
            uploaded_file = st.file_uploader("Choose a file")
        with st.sidebar.expander("Show Personal"):
            #st.subheader('Sensor Results')
            fullname = st.text_input('Give The Fullname of the Person')
            #pm = st.number_input('Give Personal Mass')
            platform_mass = st.number_input('Give Platform Mass')
            frequency = st.number_input('Give System Frequency', value=1000)
            rms_step = st.number_input("Give RMS step ", value=100, step=50)
        a=platform_mass
        @st.cache  # No need for TTL this time. It's static data :)
        def get_data():
            if platform_mass>1:
                df = pd.read_csv(uploaded_file, sep='\s+', header=None)
                
                cols = len(df.axes[1])
                if cols == 10:
                #df = pd.read_csv("data.txt", sep=" ", header=None, names=["A", "B"])
                    df.columns = ['Time', 'Col_1', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_6', 'Col_7', 'Col_8', 'Col_9']
                if cols == 11:
                    df.columns = ['Time', 'Col_1', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_6', 'Col_7', 'Col_8', 'Col_9','Col_10']
                # 'Weight_1', 'Weight_2', 'Weight_3', 'Weight_4', 'Weight_Sum',
                C = 406.831
                #sr = 1000
                resolution = 16
                # Calculate for A Sensor Mass $ Weight
                Vfs_1 = 2.00016
                df['Mass_1'] = df['Mass_1'] * C / (Vfs_1 * ( (2**resolution) - 1 ) )
                # Calculate for B Sensor Mass $ Weight
                Vfs_2 = 2.00002
                df['Mass_2'] = df['Mass_2'] * C / (Vfs_2 * ( (2**resolution) - 1 ) )
                # Calculate for C Sensor Mass $ Weight
                Vfs_3 = 2.00057
                df['Mass_3'] = df['Mass_3'] * C / (Vfs_3 * ( (2**resolution) - 1 ) )
                # Calculate for D Sensor Mass $ Weight
                Vfs_4 = 2.00024
                df['Mass_4'] = df['Mass_4'] * C / (Vfs_4 * ( (2**resolution) - 1 ) )
                # Calculate the sum of all sensors Mass $ Weight
                df['Mass_Sum'] = (df['Mass_1'] + df['Mass_2'] + df['Mass_3'] + df['Mass_4']) - platform_mass
                #df2 = df[df['col2'] < 0]
                #df[df['B'] > 10]
                #df[df['Mass_Sum'] > 100]
                pm = df['Mass_Sum'].mean()

                # Show results only over specific values
                #df = df[df['Mass_Sum'] > 0.044]

                W = 450
                L = 450

                df['CoPX'] = W * (( df['Mass_3'] + df['Mass_2'] - df['Mass_1'] - df['Mass_4'] )) / 2 * ( df['Mass_3'] + df['Mass_2'] + df['Mass_1'] + df['Mass_4'] )
                df['CoPY'] = L * (( df['Mass_2'] + df['Mass_1'] - df['Mass_3'] - df['Mass_4'] )) / 2 * ( df['Mass_3'] + df['Mass_2'] + df['Mass_1'] + df['Mass_4'] )
                df['Rows_Count'] = df.index
                return platform_mass, df[['Time', 'CoPX', 'CoPY', 'Rows_Count']]
        if a > 1:
            platform_mass, df = get_data()
            #Create a RMS Step Choice

            min_time = int(df.index.min())
            max_time = int(df.index.max())
            min_CoPX = min(df['CoPX'])
            max_CoPX = max(df['CoPX'])

            selected_time_range = st.sidebar.slider('Select the time range, per 100', min_time, max_time, (min_time, max_time), 100)
            df_selected_model = (df.Rows_Count.between(selected_time_range[0], selected_time_range[1]) )
            df = pd.DataFrame(df[df_selected_model])

            @st.cache
            def altair_histogram():
                brushed = alt.selection_interval(encodings=["x"], name="brushed")

                return (
                    alt.Chart(df)
                    .mark_circle(size=10)
                    .encode(alt.X("CoPX:Q"), y="CoPY:Q", tooltip=['CoPX', 'CoPY'])
                    .add_selection(brushed)
                ).properties(width=1300, height=500)

            event_dict = altair_component(altair_chart=altair_histogram())

            r = event_dict.get("CoPX")
            if r:
                filtered = df[(df.CoPX >= r[0]) & (df.CoPX < r[1])]
                df1 = pd.DataFrame(filtered)
                #pd.DataFrame(df[df_selected_model])

                st.write('The Min & Max CoPX values of this time range are:', min(df1['CoPX']),  max(df1['CoPX']))
                st.write('The Min & Max CoPY values of this time range are:', min(df1['CoPY']), max(df1['CoPY']))

                st.sidebar.write('Time range from', min(df1['Rows_Count']), 'to', max(df1['Rows_Count']), 'ms')
                st.sidebar.write('Min CoPX:', min(df1['CoPX']))
                st.sidebar.write('Max CoPX:', max(df1['CoPX']))
                st.sidebar.write('Min CoPY:',  min(df1['CoPY']))
                st.sidebar.write('Max CoPY:',  max(df1['CoPY']))

                selected_filtered_columns = st.multiselect(
                label='What column do you want to display', default=('Time', 'CoPX', 'CoPY'), help='Click to select', options=df1.columns)
                st.write(df1[selected_filtered_columns])

                st.download_button(
                    label="Export table",
                    data=df[selected_filtered_columns].to_csv(),
                    file_name='df.csv',
                    mime='text/csv',
                )

            else:
                st.write("")
                st.sidebar.write('Time range from', min(df['Rows_Count']), 'to', max(df['Rows_Count']), 'ms')
                st.sidebar.write('Min CoPX:', min(df['CoPX']))
                st.sidebar.write('Max CoPY:', max(df['CoPX']))
                st.sidebar.write('Min CoPX:',  min(df['CoPY']))
                st.sidebar.write('Max CoPY:',  max(df['CoPY']))
                st.write('The Min & Max CoPX values of this time range are:', min(df['CoPX']), max(df['CoPX']))
                st.write('The Min & Max CoPY values of this time range are:', min(df['CoPY']), max(df['CoPY']))
                selected_clear_columns = st.multiselect(
                label='What column do you want to display', default=('Time', 'CoPX', 'CoPY'), help='Click to select', options=df.columns)
                st.write(df[selected_clear_columns])
                st.download_button(
                    label="Export Table",
                    data=df[selected_clear_columns].to_csv(),
                    file_name='df.csv',
                    mime='text/csv',
                )
    
if __name__ == '__main__':
    main()