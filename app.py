import streamlit as st
import pandas as pd
import numpy as np
from streamlit_vega_lite import altair_component
import altair as alt
import biosignalsnotebooks as bsnb
import csv
import sympy as sp
from scipy.integrate import quad
# import xlsxwriter
# from io import BytesIO

# output = BytesIO()

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

#st.title(body, anchor=None)
#st.title('This is a title')
#st.subheader(body, anchor=None)
st.subheader('School of Physical Education and Sports Science1')

def main():
    page = st.sidebar.selectbox("Choose a page", ['Unconverted Results', 'Unconverted Center of pressure', 'Converted Results'])
    if page == 'Unconverted Results':
        with st.expander("Show File Form"):
            uploaded_file = st.file_uploader("Choose a file")
        with st.sidebar.expander("Show Personal"):
            #st.subheader('Sensor Results')
            fulname = st.text_input('Give The Fullname of the Person')
            #pm = st.number_input('Give Personal Mass')
            platform_mass = st.number_input('Give Platform Mass')
            frequency = st.number_input('Give System Frequency', value=1000)
            rms_step = st.number_input("Give RMS step ", value=100, step=50)
            type_of_try = st.selectbox(
                    'Select type of try', ('vertical_jump','belts'))
        a=platform_mass         
        @st.cache(allow_output_mutation=True)
        def get_data():
            if platform_mass > 1:
                df = pd.read_csv(uploaded_file, sep='\s+', header=None)
                
                cols = len(df.axes[1])
                if cols == 10:
                #df = pd.read_csv("data.txt", sep=" ", header=None, names=["A", "B"])
                    df.columns = ['Time', 'Col_1', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_6', 'Col_7', 'Col_8', 'Col_9']
                if cols == 11:
                    df.columns = ['Time', 'Col_1', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_6', 'Col_7', 'Col_8', 'Col_9','Col_10']
                #
                
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
                pm = df['Mass_Sum'].mean()
                df['Force'] = df['Mass_Sum'] * 9.81
                #Find Acceleration
                df['Acceleration'] = (df['Force'] / pm) - 9.81
                #Find Velocity
                df['Start_Velocity'] = df.Acceleration.rolling(window=2,min_periods=1).mean()*0.001
                df['Velocity'] = df.Start_Velocity.rolling(window=999999,min_periods=1).sum()
                df['Velocity1000'] = df['Velocity']*1000


                #THIS IS ALL FOR EMG TO RMS 1
                # [Baseline Removal]
                pre_pro_signal1 = df['Col_8'] - df["Col_8"].mean()
                # [Signal Filtering]
                low_cutoff = 10 # Hz
                high_cutoff = 450 # Hz
                # Application of the signal to the filter. This is EMG1 after filtering
                pre_pro_signal1= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal1, low_cutoff, high_cutoff, frequency)
                df['pre_pro_signalEMG1'] = pre_pro_signal1**2
                #This is RMS per 100
                df['RMS100_1'] = df.pre_pro_signalEMG1.rolling(window=100,min_periods=100).mean()**(1/2)



                #THIS IS ALL FOR EMG TO RMS 2
                # [Baseline Removal]
                pre_pro_signal2 = df['Col_9'] - df["Col_9"].mean()
                # [Signal Filtering]
                low_cutoff = 10 # Hz
                high_cutoff = 450 # Hz
                # Application of the signal to the filter. This is EMG1 after filtering
                pre_pro_signal2= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal2, low_cutoff, high_cutoff, frequency)
                df['pre_pro_signalEMG2'] = pre_pro_signal2**2
                #This is RMS per 100
                df['RMS100_2'] = df.pre_pro_signalEMG2.rolling(window=100,min_periods=100).mean()**(1/2)

                #THIS IS ALL FOR EMG TO RMS 3
                # [Baseline Removal]
                pre_pro_signal3 = df['Col_10'] - df["Col_10"].mean()
                # [Signal Filtering]
                low_cutoff = 10 # Hz
                high_cutoff = 450 # Hz
                # Application of the signal to the filter. This is EMG1 after filtering
                pre_pro_signal3= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal3, low_cutoff, high_cutoff, frequency)
                df['pre_pro_signalEMG3'] = pre_pro_signal3**2
                #This is RMS per 100
                df['RMS100_3'] = df.pre_pro_signalEMG3.rolling(window=100,min_periods=100).mean()**(1/2)

                #Create new column for index
                df['Rows_Count'] = df.index
                return pm, platform_mass, df
############################################################################################################                
                
        if a > 1:
            pm, platform_mass, df = get_data()
            # if rms_step >0:
            #     df['RMS100_1'] = df.pre_pro_signalEMG1.rolling(window=int(rms_step),min_periods=int(rms_step)).mean()**(1/2)
            #     df['RMS100_2'] = df.pre_pro_signalEMG2.rolling(window=int(rms_step),min_periods=int(rms_step)).mean()**(1/2)
            #     df['RMS100_3'] = df.pre_pro_signalEMG3.rolling(window=int(rms_step),min_periods=int(rms_step)).mean()**(1/2)

            #Find Maximum Velocity
            Vmax = max(df['Velocity'])

            #Find the std of 15 values and multiply with 5. Declaring the correct time for starting the Jump.
            specific_std = df.loc[500:1500:1, 'Force'].std()
            k = df.loc[500:1500:1, 'Force'].mean()
            specific_std_10 = specific_std * 10

            #Find the Time that STARTS the Try.
            if type_of_try == 'vertical_jump':
                for i in range (5, len(df.index)):
                    if df.loc[i,'Force'] < (k - specific_std_10):
                        start_try_time = i-30
                        break

            #Find the proper moment of TAKE OFF Time.
            
                closest_to_min_force = df['Force'].sub(df['Force'].min()).abs().idxmin()
            
                for i in range (start_try_time, len(df.index)):
                    if df.loc[i,'Force'] < 2:
                        take_off_time = i
                        break

                #Find the IMPULSE GRF
                df['cropped'] = df.loc[(start_try_time-1000):take_off_time:1,'Force']
                df['Impulse_grf'] = df['cropped'] * (1/1000)
                impulse_grf = df['Impulse_grf'].sum()

                #Find the IMPULSE BW
                impulse_bw_duration = (take_off_time - (start_try_time-1000)) / 1000
                impulse_bw = pm * 9.81 * impulse_bw_duration

                #Find the Velocity depending on impulse
                velocity_momentum = (impulse_grf - impulse_bw) / pm
                
                #Find the proper moment of LANDING.
                std_after_landing = df.loc[closest_to_min_force:(closest_to_min_force+15):1, 'Force'].std()
                for i in range (take_off_time, len(df.index)):
                    #if df.loc[i,'Force'] < df.loc[(i+1),'Force'] - (std_after_landing * 5):
                    if df.loc[i,'Force'] > 15:
                        landing_time = i
                        break

                    #Find the time between the TAKE OFF and landing & Take Off Velocity
                    take_off_till_landing_duration = (landing_time - take_off_time) * 0.001
                    take_off_velocity = (9.81 * take_off_till_landing_duration) / 2
                
                    #Find the Jump ot the Try depending of the TAKE OFF VELOCITY
                    jump_depending_take_off_velocity = (take_off_velocity ** 2) / (9.81 * 2)

                    #Find the Jump ot the Try depending of the IMPULSE
                    jump_depending_impluse1 = (velocity_momentum ** 2) / (9.81 * 2)

                    #Find the Time in Air
                    in_air_time = (take_off_time - start_try_time) * 0.001

                    #Find the Jump of the try depending on Time in Air
                    Jump_depending_in_air_time = (1/2) * 9.81 * ((in_air_time / 2) ** (1/2))

                    #Find The Closest to Zero Velocity
                    dfv = df[(df.index >= start_try_time) & (df.index <start_try_time + 500)]            
                    closest_zero_velocity = dfv['Velocity'].sub(0).abs().idxmin()

                #Define The Whole Time Range Of Graph
                min_time = int(df.index.min())
                max_time = int(df.index.max())
                with st.expander("Time Range"):
                    selected_time_range = st.slider('Select the whole time range of the graph, per 100', min_time, max_time, (min_time, max_time), 100)
                df_selected_model = (df.Rows_Count.between(selected_time_range[0], selected_time_range[1]) )
                df = pd.DataFrame(df[df_selected_model])
                #Values Sidebar
                with st.sidebar.expander(("Values"), expanded=True):
                    st.write('Body mass is:', round(pm,2), 'kg')
                    st.write('Platform mass is:', round(platform_mass,2), 'kg')
                    st.write('Try starts at:', start_try_time, 'ms')
                    st.write('Take Off Time starts at:', take_off_time, 'ms')
                    st.write('Landing Time starts at:', landing_time, 'ms')
                    st.write('Impulse (GRF) is:', round(impulse_grf,4), 'N/s')
                    st.write('Impulse (BW) is:', round(impulse_bw,4), 'N/s')
                    st.write('Net Impulse is:', round((impulse_grf - impulse_bw),4), 'N/s')
                    st.write('Jump (Take Off Velocity) is:', round(jump_depending_take_off_velocity, 4), 'm')
                    st.write('Jump (Impulse) is:', round(jump_depending_impluse1, 4), 'm')
                #Calculate RFD & R_EMG
                with st.sidebar.expander("RFD & EMG"):
                    if closest_zero_velocity > 1:
                        st.write("Closest to Zero Velocity is in:",closest_zero_velocity, "ms")
                        closest_zero_velocity_start = closest_zero_velocity
                        closest_zero_velocity_finish = closest_zero_velocity + 500
                        user_time_input_min = st.number_input("Time From ",  value=int(closest_zero_velocity_start))
                        user_time_input_max = st.number_input("Till Time ", value=int(closest_zero_velocity_finish))
                        dfRFD = df[(df.index >= user_time_input_min) & (df.index < user_time_input_max)]     
                #Create Graph
                with st.expander("Graph Velocity-Force-RMS", expanded=True):
                    brushed = alt.selection_interval(encodings=['x'], name="brushed")
                    base = alt.Chart(df).mark_line().transform_fold(
                        ['Velocity1000', 'Force','RMS100_1, RMS100_2, RMS100_3'],
                        as_=['Measure', 'Value']
                    ).encode(alt.Color('Measure:N'),alt.X('Rows_Count:T'),tooltip=['Rows_Count', 'Force', 'Velocity', 'Force', 'Acceleration', 'RMS100_1, RMS100_2, RMS100_3'])
                    line_A = base.transform_filter(
                        alt.datum.Measure == 'Velocity'
                    ).encode(
                        alt.Y('average(Value):Q', axis=alt.Axis(title='Velocity')),
                    )
                    line_B = base.transform_filter(
                        alt.datum.Measure == 'Force'
                    ).encode(
                        alt.Y('Value:Q',axis=alt.Axis(title='Force')),
                    )
                    line_C = base.transform_filter(
                        alt.datum.Measure == 'RMS100_1'
                    ).encode(
                        alt.Y('Value:Q',axis=alt.Axis(labelPadding= 50, title='RMS100_1'))
                    )
                    line_D = base.transform_filter(
                        alt.datum.Measure == 'RMS100_2'
                    ).encode(
                        alt.Y('Value:Q',axis=alt.Axis(labelPadding= 50, title='RMS100_2'))
                    )
                    line_E = base.transform_filter(
                        alt.datum.Measure == 'RMS100_3'
                    ).encode(
                        alt.Y('Value:Q',axis=alt.Axis(labelPadding= 50, title='RMS100_3'))
                    )
                    #c=alt.layer(line_A, line_B, line_C).resolve_scale(y='independent').properties(width=950)
                    #Display Chart
                    #st.altair_chart(c, use_container_width=True)
                
                    @st.cache(allow_output_mutation=True)
                    def altair_histogram():
                        brushed = alt.selection_interval(encodings=["x"], name="brushed")
                        # on="[mousedown[!event.shiftKey], mouseup] > mousemove",
                        # translate="[mousedown[!event.shiftKey], mouseup] > mousemove!",                     
                        return (
                            alt.Chart(df).transform_fold(
                                ['Velocity1000', 'Force', 'RMS100_1', 'RMS100_2', 'RMS100_3']
                            ).resolve_scale(y='independent')
                            .mark_line().resolve_scale(y='independent')
                            .encode(alt.X("Rows_Count:Q"), y="value:Q", tooltip=['Rows_Count', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS100_1', 'RMS100_2', 'RMS100_3'], color='key:N').add_selection(
                                brushed
                            ).resolve_scale(y='independent')
                        ).properties(width=1000).resolve_scale(y='independent')
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
                    df['cropped1'] = df.loc[user_time_input_min_main_table:user_time_input_max_main_table:1,'Force']
                    df['Impulse_grf1'] = df['cropped1'] * (1/1000)
                    impulse_grf1 = df['Impulse_grf1'].sum()
                    #Find the IMPULSE BW
                    impulse_bw_duration1 = (user_time_input_max_main_table - user_time_input_min_main_table) / 1000
                    impulse_bw1 = pm * 9.81 * impulse_bw_duration1
                    velocity_momentum1 = (impulse_grf1 - impulse_bw1) / pm
                    jump_depending_impluse = (velocity_momentum1 ** 2) / (9.81 * 2)

                    #Find the IMPULSE GRF
                    df['cropped'] = df.loc[(start_try_time-1000):take_off_time:1,'Force']
                    df['Impulse_grf'] = df['cropped'] * (1/1000)
                    impulse_grf = df['Impulse_grf'].sum()

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
                        b_rfd1 = (X*Y).sum() / (X ** 2).sum()
                        #st.write(round(b_rfd),4)
                        headers_list_rfd1.append("RFD-"+str(i))
                        l_rfd1.append(b_rfd1)
                        #FOR EMG
                        X = df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'] - df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'].mean()
                        Y1 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG1'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG1'].mean()
                        Y2 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG2'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG2'].mean()
                        Y3 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG3'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG3'].mean()

                        
                        b_emg1 = (X*Y1).sum() / (X ** 2).sum()
                        b_emg2 = (X*Y2).sum() / (X ** 2).sum()
                        b_emg3 = (X*Y3).sum() / (X ** 2).sum()

                        headers_list_emg1.append("EMG1-"+str(i))
                        headers_list_emg2.append("EMG2-"+str(i))
                        headers_list_emg3.append("EMG3-"+str(i))
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

                    #Dataframe for EMG
                    if emg_df1.empty:
                        emg_df1 = pd.DataFrame([l_emg1])
                        cols = len(emg_df1.axes[1])
                        emg_df1.columns = [*headers_list_emg1]
                    else:
                        to_append = emg_df1
                        emg_df1_length = len(emg_df1)
                        emg_df1.loc[emg_df1_length] = to_append
                    
                    #Dataframe for EMG
                    if emg_df2.empty:
                        emg_df2 = pd.DataFrame([l_emg2])
                        cols = len(emg_df2.axes[1])
                        emg_df2.columns = [*headers_list_emg2]
                    else:
                        to_append = emg_df2
                        emg_df2_length = len(emg_df2)
                        emg_df2.loc[emg_df2_length] = to_append

                    #Dataframe for EMG
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
                                st.write('Impulse GRF:', round(impulse_grf1,4))
                                st.write('Impulse BW:', round(impulse_bw1,4))
                                st.write('Net Impulse:', round(impulse_grf1 - impulse_bw1,4))
                                #st.write('velocity_momentum:', round(velocity_momentum1,2))
                                st.write('Jump (Impluse):', round(jump_depending_impluse,4))
                        with col2:
                                st.write('Force-Mean:', round(df_brushed["Force"].mean(),4))
                                st.write('Force-Min:', round(min(df_brushed['Force']),4))
                                st.write('Force-Max:', round(max(df_brushed['Force']),4))
                        with col3:
                                st.write('RMS100_1-Mean:', round(df_brushed["RMS100_1"].mean(),4))
                                st.write('RMS100_2-Mean:', round(df_brushed['RMS100_2'].mean(),4))
                                st.write('RMS100_3-Mean:', round(df_brushed['RMS100_3'].mean(),4))
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
                        label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS100_1', 'RMS100_2', 'RMS100_3'), help='Click to select', options=df_brushed.columns)
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
                    
                            'Fullname' : [fulname],
                            'Type of try' : [type_of_try],
                            'Body Mass (kg)': [pm],
                            'Platform Mass (kg)': [platform_mass],
                            'Jump (Velocity Take Off) (m/s)' : [jump_depending_impluse1],
                            'Take Off Time (s)' : [take_off_time],
                            'Landing Time (s)' : [landing_time],
                            'Impulse (GRF) (N/s)' : [impulse_grf],
                            'Impulse (BW) (N/s)' : [impulse_bw],
                            'RMS_1 Mean' : [df_brushed['RMS100_1'].mean()],
                            'RMS_2 Mean' : [df_brushed['RMS100_2'].mean()],
                            'RMS_3 Mean' : [df_brushed['RMS100_3'].mean()],
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
                    # workbook = xlsxwriter.Workbook(output, {'in_memory': True})
                    # worksheet = workbook.add_worksheet()

                    # worksheet.write(final_results_df, final_results_df.columns)


                    # workbook.close()

                    # writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')

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
                                st.write('Force-Mean:', df["Force"].mean())
                                st.write('Force-Min:', min(df['Force']))
                                st.write('Force-Max:', max(df['Force']))
                        with col2:
                                st.write('Mass-Mean:', df["Mass_Sum"].mean())
                                st.write('Mass-Min:', min(df['Mass_Sum']))
                                st.write('Mass-Max:', max(df['Mass_Sum']))
                        with col3:
                                st.write('Velocity-Mean:', df["Velocity"].mean())
                                st.write('Velocity-Min:', min(df['Velocity']))
                                st.write('Velocity-Max:', max(df['Velocity']))
                        with col4:
                                st.write('RMS100_1-Mean:', df["RMS100_1"].mean())
                                st.write('RMS100_2-Mean:', df["RMS100_2"].mean())
                                st.write('RMS100_3-Mean:', df["RMS100_3"].mean())
                        with col5:
                                st.write('Acceleration-Mean:', df["Acceleration"].mean())
                                st.write('Acceleration-Min:', min(df['Acceleration']))
                                st.write('Acceleration-Max:', max(df['Acceleration']))
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
                        label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS100_1','RMS100_2', 'RMS100_3'), help='Click to select', options=df.columns)
                        st.write(df[selected_clear_columns])
                        #Button to export results
                        st.download_button(
                            label="Export table",
                            data=df[selected_clear_columns].to_csv(),
                            file_name='df.csv',
                            mime='text/csv',
                        )
            ######################################### BELT ##################################################################
            if type_of_try == 'belts':
                @st.cache(allow_output_mutation=True)
                def altair_histogram():
                    brushed = alt.selection_interval(encodings=["x"], name="brushed")
                    return (
                        alt.Chart(df).transform_fold(
                            ['Force', 'RMS100_1', 'RMS100_2', 'RMS100_3']
                        ).resolve_scale(y='independent')
                        .mark_line().resolve_scale(y='independent')
                        .encode(alt.X("Rows_Count:Q"), y="value:Q", tooltip=['Rows_Count', 'Force', 'Mass_Sum', 'RMS100_1', 'RMS100_2', 'RMS100_3'], color='key:N').add_selection(
                            brushed
                        ).resolve_scale(y='independent')
                    ).properties(width=1000).resolve_scale(y='independent')
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
                        b_rfd1 = (X*Y).sum() / (X ** 2).sum()
                        #st.write(round(b_rfd),4)
                        headers_list_rfd1.append("RFD-"+str(i))
                        l_rfd1.append(b_rfd1)
                        #FOR EMG
                        X = df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'] - df_brushed.loc[user_time_input_min_main_table:i:1,'Rows_Count'].mean()
                        Y1 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG1'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG1'].mean()
                        Y2 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG2'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG2'].mean()
                        Y3 = df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG3'] - df_brushed.loc[user_time_input_min_main_table:i:1,'pre_pro_signalEMG3'].mean()

                        
                        b_emg1 = (X*Y1).sum() / (X ** 2).sum()
                        b_emg2 = (X*Y2).sum() / (X ** 2).sum()
                        b_emg3 = (X*Y3).sum() / (X ** 2).sum()

                        headers_list_emg1.append("EMG1-"+str(i))
                        headers_list_emg2.append("EMG2-"+str(i))
                        headers_list_emg3.append("EMG3-"+str(i))
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

                    #Dataframe for EMG
                    if emg_df1.empty:
                        emg_df1 = pd.DataFrame([l_emg1])
                        cols = len(emg_df1.axes[1])
                        emg_df1.columns = [*headers_list_emg1]
                    else:
                        to_append = emg_df1
                        emg_df1_length = len(emg_df1)
                        emg_df1.loc[emg_df1_length] = to_append
                    
                    #Dataframe for EMG
                    if emg_df2.empty:
                        emg_df2 = pd.DataFrame([l_emg2])
                        cols = len(emg_df2.axes[1])
                        emg_df2.columns = [*headers_list_emg2]
                    else:
                        to_append = emg_df2
                        emg_df2_length = len(emg_df2)
                        emg_df2.loc[emg_df2_length] = to_append

                    #Dataframe for EMG
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
                                st.write('RMS100_1-Mean:', round(df_brushed["RMS100_1"].mean(),4))
                                st.write('RMS100_2-Mean:', round(df_brushed['RMS100_2'].mean(),4))
                                st.write('RMS100_3-Mean:', round(df_brushed['RMS100_3'].mean(),4))
                        
                    
                    #Display Dataframe in Datatable
                    with st.expander("Show Data Table", expanded=True):
                        selected_filtered_columns = st.multiselect(
                        label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum',  'RMS100_1', 'RMS100_2', 'RMS100_3'), help='Click to select', options=df_brushed.columns)
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
                    
                            'Fullname' : [fulname],
                            'Type of try' : [type_of_try],
                            'Body Mass (kg)': [pm],
                            'Platform Mass (kg)': [platform_mass],
                            'RMS_1 Mean' : [df_brushed['RMS100_1'].mean()],
                            'RMS_2 Mean' : [df_brushed['RMS100_2'].mean()],
                            'RMS_3 Mean' : [df_brushed['RMS100_3'].mean()],
                            'Force Mean (N)' : [df_brushed['Force'].mean()],
                            'Force Max (N)' : [max(df_brushed['Force'])],
                            'Force Min (N)' : [min(df_brushed['Force'])],
                            }
                    
                    
                    specific_metrics_df = pd.DataFrame(specific_metrics)
                    #specific_metrics_df = specific_metrics_df.round(decimals = 2)

                    #Combine all dataframes to one , for the final export
                    final_results_df = pd.concat([specific_metrics_df, rfd_df1, emg_df1, emg_df2, emg_df3], axis=1, join='inner')
                    #final_results_df['Body Mass (kg)'] = final_results_df['Body Mass (kg)'].round(decimals = 2)
                    final_results_df =np.round(final_results_df, decimals = 4)
                    # workbook = xlsxwriter.Workbook(output, {'in_memory': True})
                    # worksheet = workbook.add_worksheet()

                    # worksheet.write(final_results_df, final_results_df.columns)


                    # workbook.close()

                    # writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')

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
                                st.write('RMS100_1-Mean:', df["RMS100_1"].mean())
                                st.write('RMS100_2-Mean:', df["RMS100_2"].mean())
                                st.write('RMS100_3-Mean:', df["RMS100_3"].mean())
                        
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
                        label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS100_1','RMS100_2', 'RMS100_3'), help='Click to select', options=df.columns)
                        st.write(df[selected_clear_columns])
                        #Button to export results
                        st.download_button(
                            label="Export table",
                            data=df[selected_clear_columns].to_csv(),
                            file_name='df.csv',
                            mime='text/csv',
                        )
            
            

##############################################################################################################################################
    elif page == 'Unconverted Center of pressure':
        with st.expander("Show File Form"):
            uploaded_file = st.file_uploader("Choose a file")
        with st.sidebar.expander("Show Personal"):
            #st.subheader('Sensor Results')
            fulname = st.text_input('Give The Fullname of the Person')
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
    # HERE stars the Converted Results Values 
    else:
        with st.expander("Show Form"):
            uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            #st.subheader('Sensor Results')
            col1, col2, col3 = st.columns(3)
            with col1:
                pm = st.number_input('Give Personal Mass')
            with col2:
                platform_mass = st.number_input('Give Platform Mass')
            with col3:
                frequency = st.number_input('Give System Frequency', value=1000)

            @st.cache(allow_output_mutation=True)
            def get_data():
                df = pd.read_csv(uploaded_file, sep='\s+', header=None)
                #df = pd.read_csv("data.txt", sep=" ", header=None, names=["A", "B"])

                df.columns = ['Time', 'Col_1', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_6', 'Col_7', 'Col_8', 'Col_9']
                # C = 406.831
                # #sr = 1000
                # resolution = 16
                # # Calculate for A Sensor Mass $ Weight
                # Vfs_1 = 2.00016
                # df['Mass_1'] = df['Mass_1'] * C / (Vfs_1 * ( (2**resolution) - 1 ) )
                # # Calculate for B Sensor Mass $ Weight
                # Vfs_2 = 2.00002
                # df['Mass_2'] = df['Mass_2'] * C / (Vfs_2 * ( (2**resolution) - 1 ) )
                # # Calculate for C Sensor Mass $ Weight
                # Vfs_3 = 2.00057
                # df['Mass_3'] = df['Mass_3'] * C / (Vfs_3 * ( (2**resolution) - 1 ) )
                # # Calculate for D Sensor Mass $ Weight
                # Vfs_4 = 2.00024
                # df['Mass_4'] = df['Mass_4'] * C / (Vfs_4 * ( (2**resolution) - 1 ) )
                 # Calculate the sum of all sensors Mass $ Weight
                df['Mass_Sum'] = ((df['Mass_1'] + df['Mass_2'] + df['Mass_3'] + df['Mass_4']) / 2) - platform_mass
                #df['Mass_Sum'] = ((df['Mass_1'] + df['Mass_2'] + df['Mass_3'] + df['Mass_4']) ) / 2  - platform_mass
                df['Force'] = df['Mass_Sum'] * 9.81
                df['Acceleration'] = (df['Force'] / pm ) - 9.81
                # V1 = V0 + (a0/a1)mean *0.0001
                df['Start_Velocity'] = df.Acceleration.rolling(window=2,min_periods=1).mean()*0.001
                df['Velocity'] = df.Start_Velocity.rolling(window=999999,min_periods=1).sum()

                df['Velocity cm/s'] = df['Velocity']/100
                #df['EMG'] = (df['Col_8'] **2 )*100
                #df['EMG1'] = (df['EMG'])/100
                #df['RMS'] = df.EMG.rolling(window=50,min_periods=50).mean()**(1/2)

                # [Baseline Removal]
                pre_pro_signal1= df['Col_8'] - df["Col_8"].mean()
                # [Signal Filtering]
                low_cutoff = 10 # Hz
                high_cutoff = 450 # Hz
                # Application of the signal to the filter. This is EMG after filtering
                pre_pro_signal1= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal, low_cutoff, high_cutoff, frequency)
                df['pre_pro_signalEMG1'] = pre_pro_signal1**2
                #This is RMS per 100
                #df['RMS50'] = df.pre_pro_signalEMG1.rolling(window=50,min_periods=50).mean()**(1/2)
                df['RMS100_1'] = df.pre_pro_signalEMG1.rolling(window=100,min_periods=100).mean()**(1/2)

                df['Rows_Count'] = df.index


                return df
            df = get_data()
            rms_step = st.number_input("Give RMS step ", value=100)
            if rms_step >0:
                df['RMS100_1'] = df.pre_pro_signalEMG1.rolling(window=int(rms_step),min_periods=int(rms_step)).mean()**(1/2)

            

            st.write(jump, Vmax)

            #for i in range(0, len(df['pre_pro_signal2']),50):
            #df['RMSmean100'] = (df['pre_pro_signal2'].mean())**(1/2)


            #Find The Closest to Zero Velocity
            closest_zero_velocity = df['Velocity'].sub(0).abs().idxmin()
            #Define The Main Time Range Of Graph
            min_time = int(df.index.min())
            max_time = int(df.index.max())
            selected_time_range = st.sidebar.slider('Select the whole time range of the graph, per 100', min_time, max_time, (min_time, max_time), 100)
            df_selected_model = (df.Rows_Count.between(selected_time_range[0], selected_time_range[1]) )
            df = pd.DataFrame(df[df_selected_model])
            #Calculate RFD
            with st.sidebar.expander("RFD & EMG"):
                if closest_zero_velocity > 1:
                    st.write("Closest to Zero Velocity is in:",closest_zero_velocity, "ms")
                    closest_zero_velocity_start = closest_zero_velocity
                    closest_zero_velocity_finish = closest_zero_velocity + 500
                    user_time_input_min = st.number_input("Time From ",  value=int(closest_zero_velocity_start))
                    user_time_input_max = st.number_input("Till Time ", value=int(closest_zero_velocity_finish))
                    dfRFD = df[(df.index >= user_time_input_min) & (df.index < user_time_input_max)]


            with st.expander("Graph Velocity-Force-RMS", expanded=True):
                #alt.data_transformers.enable('json')
                # @st.cache(allow_output_mutation=True)
                # def chart():
                #rms_step = ['RMS50','RMS100_1']
                brushed = alt.selection_interval(encodings=['x'], name="brushed")
                base = alt.Chart(df).mark_line().transform_fold(
                    ['Velocity', 'Force','RMS100_1'],
                    as_=['Measure', 'Value']
                ).encode(alt.Color('Measure:N'),alt.X('Rows_Count:T'),tooltip=['Rows_Count', 'Force', 'Velocity', 'Force', 'Acceleration', 'RMS100_1'])
                line_A = base.transform_filter(
                    alt.datum.Measure == 'Velocity'
                ).encode(
                    alt.Y('average(Value):Q', axis=alt.Axis(title='Velocity')),
                )
                line_B = base.transform_filter(
                    alt.datum.Measure == 'Force'
                ).encode(
                    alt.Y('Value:Q',axis=alt.Axis(title='Force')),
                )
                line_C = base.transform_filter(
                    alt.datum.Measure == 'RMS100_1'
                ).encode(
                    alt.Y('Value:Q',axis=alt.Axis(labelPadding= 50, title='RMS100_1'))
                )
                c=alt.layer(line_A, line_B, line_C).resolve_scale(y='independent').properties(width=950)

                  #  return c
                #l = chart()
                # event_dict1 = altair_component(c)
                # p = event_dict1.get("Rows_Count")
                # st.write(p)
                # st.write(event_dict1)
                # if p is not None:
                #     st.write('eksw')
                # else:
                #     st.write('mesa')
                # selected = base.transform_filter(brushed).mark_area(color='goldenrod')
                # alt.layer(base.add_selection(brushed), selected)
                # #c = chart()
                # st.altair_chart(c, use_container_width=True)
            
            # zoom = alt.selection_interval(
            #     bind='scales',
            #     on="[mousedown[!event.shiftKey], mouseup] > mousemove",
            #     translate="[mousedown[!event.shiftKey], mouseup] > mousemove!",
            # )

            # selection = alt.selection_interval(
            #     on="[mousedown[event.shiftKey], mouseup] > mousemove",
            #     translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
            # )

            # alt.Chart(source).mark_circle(size=60).encode(
            #     x='Horsepower',
            #     y='Miles_per_Gallon',
            #     color='Origin',
            # ).add_selection(zoom, selection)
            

            with st.expander("Graph Velocity-Force-RMS.", expanded=True):
                @st.cache(allow_output_mutation=True)
                def altair_histogram():
                    zoom = alt.selection_interval(
                    bind='scales',
                    on="[mousedown[event.shiftKey], mouseup] > mousemove",
                    translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
            )
                    brushed = alt.selection_interval(on="[mousedown[!event.shiftKey], mouseup] > mousemove",
                                                     translate="[mousedown[!event.shiftKey], mouseup] > mousemove!", 
                                                     encodings=['x'], name="brushed")
                    return (
                        alt.Chart(df).transform_fold(
                            ['Velocity', 'Force','RMS100_1']
                        ).resolve_scale(y='independent')
                        .mark_line().resolve_scale(y='independent')
                        .encode(alt.X("Rows_Count:Q"), y="value:Q", tooltip=['Rows_Count', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS100_1'], color='key:N').add_selection(
                            zoom, brushed
                        ).resolve_scale(y='independent')

                        #.add_selection(brushed)
                    ).properties(width=1000).resolve_scale(y='independent')

                event_dict = altair_component(altair_chart=altair_histogram())
                r = event_dict.get("Rows_Count")
            #st.write(r)
            #st.write(event_dict)
            col1, col2 = st.columns(2)
            with col1:
                if r:
                    if isinstance(r[0], float) is True:
                        t = int(r[0])
                        user_time_input_min_main_table = st.number_input("From Time ",value=t)
                    else:
                        user_time_input_min_main_table = st.number_input("From Time ",value=r[0])

                else:
                    user_time_input_min_main_table = st.number_input("From Time. ",value=0)

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
                    user_time_input_max_main_table = st.number_input("Till Time. ",value=0)




            df_brushed = df[(df.index >= user_time_input_min_main_table) & (df.index < user_time_input_max_main_table)]
            
            #df_brushed["Force"].std()
            st.write('mikos', len(df), Vmax)
            
            mean_std_5 = df.loc[7000:7015:1, 'Force'].std() *5
            mean_subtraction = df.loc[7000:7015:1, 'Force'].mean() - mean_std_5

            st.write(mean_std_5, mean_subtraction)
            #st.write(df.loc[4670, 'Force'])
            #st.write(df.loc[5000, 'Force']-e)
            #jumb_time = df['Force'] - e
            #st.write(e, j)
            #result = df_brushed["Force"].mean() - s

            #df_brushed.loc[10, Force)

            #st.write('assaasd', s , df_brushed.loc[int(user_time_input_max_main_table-1), 'Force'])
            #df_brushed.loc['5', 'Force']
            #df_brushed.loc[1, 'Force']

            #closest = df_brushed['Force'].sub(s).abs().idxmin()
            #st.write(closest)


            for i in range(4000,len(df)):
                if df.loc[i, 'Force'] <  mean_subtraction:
                    st.write('TWRA',i)
                    break
                

            if len(df_brushed):
                df_brushed = df[(df.index >= user_time_input_min_main_table) & (df.index < user_time_input_max_main_table)]
                # if r:
                #     filtered = df[(df.Rows_Count >= r[0]) & (df.Rows_Count < r[1])]
                #     df_brushed = pd.DataFrame(filtered)


                with st.expander('Show Specific Calculations', expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                            st.write('Force-Mean:', df_brushed["Force"].mean())
                            st.write('Force-STD:', df_brushed["Force"].std())
                            st.write('Force-Min:', min(df_brushed['Force']))
                            st.write('Force-Max:', max(df_brushed['Force']))
                    with col2:
                            st.write('Mass-Mean:', df_brushed["Mass_Sum"].mean())
                            st.write('Mass-Min:', min(df_brushed['Mass_Sum']))
                            st.write('Mass-Max:', max(df_brushed['Mass_Sum']))
                    with col3:
                            st.write('Velocity-Mean:', df_brushed["Velocity"].mean())
                            st.write('Velocity-Min:', min(df_brushed['Velocity']))
                            st.write('Velocity-Max:', max(df_brushed['Velocity']))
                    with col4:
                            st.write('Acceleration-Mean:', df_brushed["Acceleration"].mean())
                            st.write('Acceleration-Min:', min(df_brushed['Acceleration']))
                            st.write('Acceleration-Max:', max(df_brushed['Acceleration']))
                    data = {'Unit': ['Force', 'Mass_Sum', 'Velocity', 'Acceleration'],
                            'Mean': [df_brushed["Force"].mean(), df_brushed["Mass_Sum"].mean(), df_brushed["Velocity"].mean(), df_brushed["Acceleration"].mean()],
                            'Min': [min(df_brushed['Force']), min(df_brushed['Mass_Sum']), min(df_brushed['Velocity']), min(df_brushed['Acceleration'])],
                            'Max': [max(df_brushed['Force']), max(df_brushed['Mass_Sum']), max(df_brushed['Velocity']), max(df_brushed['Acceleration'])],
                            #'Max': [max(df_brushed['Force']), max(df_brushed['Mass_Sum']), max(df_brushed['Velocity']), max(df_brushed['Acceleration'])] }
 }

                    df_calcs = pd.DataFrame(data)

                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                            title = st.text_input(label='Give the filename',value='filename.csv', placeholder="write here your prefer filename .csv")
                    with col2:
                            st.write(' ')
                    with col3:
                            st.write(' ')
                    with col4:
                            st.write(' ')
                    with col5:
                            st.write(' ')
                    with col6:
                            st.write(' ')

                    st.download_button(
                        label="Export metrics",
                        data=df_calcs.to_csv(),

                        file_name='specifi_metrics.csv',
                        mime='csv',
                    )
                with st.expander("Show Data Table", expanded=True):

                    selected_filtered_columns = st.multiselect(
                    label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS100_1'), help='Click to select', options=df_brushed.columns)
                    st.write(df_brushed[selected_filtered_columns])
                    #Button to export results
                    st.download_button(
                        label="Export table",
                        data=df_brushed[selected_filtered_columns].to_csv(),
                        file_name='df.csv',
                        mime='text/csv',
                    )

            else:
                slider = alt.binding_range(min=0, max=100, step=1, name='cutoff:')
                with st.expander("Show Specific Calculations", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                            st.write('Force-Mean:', df["Force"].mean())
                            st.write('Force-Min:', min(df['Force']))
                            st.write('Force-Max:', max(df['Force']))
                    with col2:
                            st.write('Mass-Mean:', df["Mass_Sum"].mean())
                            st.write('Mass-Min:', min(df['Mass_Sum']))
                            st.write('Mass-Max:', max(df['Mass_Sum']))
                    with col3:
                            st.write('Velocity-Mean:', df["Velocity"].mean())
                            st.write('Velocity-Min:', min(df['Velocity']))
                            st.write('Velocity-Max:', max(df['Velocity']))
                    with col4:
                            st.write('Acceleration-Mean:', df["Acceleration"].mean())
                            st.write('Acceleration-Min:', min(df['Acceleration']))
                            st.write('Acceleration-Max:', max(df['Acceleration']))
                data = {'Unit': ['Force', 'Mass_Sum', 'Velocity', 'Acceleration'],
                            'Mean': [df["Force"].mean(), df["Mass_Sum"].mean(), df["Velocity"].mean(), df["Acceleration"].mean()],
                            'Min': [min(df['Force']), min(df['Mass_Sum']), min(df['Velocity']), min(df['Acceleration'])],
                            'Max': [max(df['Force']), max(df['Mass_Sum']), max(df['Velocity']), max(df['Acceleration'])],
                            #'Max': [max(df_brushed['Force']), max(df_brushed['Mass_Sum']), max(df_brushed['Velocity']), max(df_brushed['Acceleration'])] }
 }
                df_calcs = pd.DataFrame(data)
                #st.write(df_calcs)
                #data = {'Name': ['Tom', 'Joseph', 'Krish', 'John'], 'Age': [20, 21, 19, 18]}
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                        title = st.text_input(label='Give the filename',value='filename.csv', placeholder="write here your prefer filename .csv")
                with col2:
                        st.write(' ')
                with col3:
                        st.write(' ')
                with col4:
                        st.write(' ')
                with col5:
                        st.write(' ')
                with col6:
                        st.write(' ')
                #st.write(title)
                #st.write(type(title))
                st.download_button(
                    label="Export metrics",
                    data=df_calcs.to_csv(),

                    file_name='specific_metrics.csv',
                    mime='csv',
                )
                st.sidebar.write('Time range from', min(df['Rows_Count']), 'to', max(df['Rows_Count']), 'ms')
                st.sidebar.write('Min Mass_Sum:', min(df['Mass_Sum']))
                st.sidebar.write('Max Mass_Sum:',  max(df['Mass_Sum']))
                #st.write('The Min & Max Weight_Sum values of this time range are:', min(df['Mass_Sum']), max(df['Mass_Sum']) )
                #st.write('The Min & Max Velocity values of this time range are:', min(df['Velocity']), max(df['Velocity']))
                with st.expander("Show Data Table", expanded=True):
                    selected_clear_columns = st.multiselect(
                    label='What column do you want to display', default=('Time', 'Force', 'Mass_Sum', 'Velocity', 'Acceleration', 'RMS100_1'), help='Click to select', options=df.columns)
                    st.write(df[selected_clear_columns])
                    #Button to export results
                    st.download_button(
                        label="Export table",
                        data=df[selected_clear_columns].to_csv(),
                        file_name='df.csv',
                        mime='text/csv',
                    )




if __name__ == '__main__':
    main()