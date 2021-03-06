import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os
import datetime
import joblib

def load_npy(name):
    return list(np.load('npy/%s.npy'%name,allow_pickle=True))

def predict():
    def devide(data, name):
        data[name] = data[name].fillna('unknown')
        all = load_npy(name + '_devided')
        for i in all:
            data[name + '_' + str(i)] = data[name].map(lambda x: 1 if i in x else 0)
        data.drop(columns=[name], inplace=True)

    def devide_time(data, name):
        print(data[name][0])
        data[name + '_year'] = data[name][0].strftime("%Y")
        data[name + '_month'] = data[name][0].strftime("%m")
        data[name + '_day'] = data[name][0].strftime("%d")
        data.drop(columns=[name], inplace=True)

    def one_hot(data, name):
        temp = load_npy(name + '_onehot')
        for i in temp:
            data[name + '_' + str(i)] = data[name].map(lambda x: 1 if x == i else 0)
        data.drop(columns=[name], inplace=True)
    def oneness(data):
        s1=', '
        s=str(data[0])
        for i in range(len(data)):
            s=s+s1+data[i]
        return [s]

    data = pd.DataFrame()
    data['Type'] = oneness(Type)
    data['Year built'] = Year_built
    data['Heating'] = oneness(Heating)
    data['Cooling'] = oneness(Cooling)
    data['Parking'] = oneness(Parking)
    data['Lot'] = Lot
    data['Bedrooms'] = oneness(Bedrooms)
    data['Bathrooms'] = Bathrooms
    data['Full bathrooms'] = Full_bathrooms
    data['Total interior livable area'] = Total_interior_livable_area
    data['Total spaces'] = Total_spaces
    data['Garage spaces'] = Garage_spaces
    data['Region'] = Region
    data['Elementary School Score'] = Elementary_School_Score
    data['Elementary School Distance'] = Elementary_School_Distance
    data['Middle School Score'] = Middle_School_Score
    data['Middle School Distance'] = Middle_School_Distance
    data['High School Score'] = High_School_Score
    data['High School Distance'] = High_School_Distance
    data['Flooring'] = oneness(Flooring)
    data['Appliances included'] = oneness(Appliances_included)
    data['Laundry features'] = oneness(Laundry_features)
    data['Parking features'] = oneness(Parking_features)
    data['Tax assessed value'] = Tax_assessed_value
    data['Annual tax amount'] = Annual_tax_amount
    data['Listed On'] = Listed_On
    data['Listed Price'] = Listed_Price
    data['Last Sold On'] = Last_Sold_On
    data['Last Sold Price'] = Last_Sold_Price
    data['Zip'] = Zip
    data['State'] = State

    devide(data, 'Type')
    devide(data, 'Heating')
    devide(data, 'Cooling')
    devide(data, 'Parking')
    devide(data, 'Bedrooms')
    devide(data, 'Flooring')
    devide(data, 'Appliances included')
    devide(data, 'Laundry features')
    devide(data, 'Parking features')

    devide_time(data, 'Listed On')
    devide_time(data, 'Last Sold On')

    one_hot(data, 'Region')
    one_hot(data, 'State')

    data['Zip_National_Area'] = data['Zip'].map(lambda x: int(str(x)[0]))
    data['Zip_Sectional_Center'] = data['Zip'].map(lambda x: int(str(x)[1:3]))
    data['Zip_Delivery_Area'] = data['Zip'].map(lambda x: int(str(x)[3:]))
    data.drop(columns=['Zip'], inplace=True)

    clf = joblib.load('model.pkl')
    y_pred = clf.predict(data)

    # result['text'] = '%.2f' % y_pred[0]
    return(y_pred[0])

# ?????????????????????????????????????????????
st.set_page_config(
    page_title="????????????",
    layout="wide"
)
# ?????????????????????????????????
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# ???????????????
sidebar = st.sidebar.radio(
    "?????????",
    ("??????", "???????????????", "????????????", "????????????")
)
if sidebar == "???????????????":
    st.title("???????????????")
    # ???????????????
    project_name = st.selectbox(
        "???????????????",
        ["?????????????????????", "?????????????????????"]
    )
    if project_name=="?????????????????????":
        df=pd.read_csv('data/train.csv')
        df1=df[:30000]
        df2=df[30000:]
        if st.checkbox('??????'):
            st.write(df1)
            st.write(df2)
        st.subheader('????????????')
        left, right = st.columns(2)
        with left:
            column = [i for i in df.columns]
            option = st.selectbox(
                "??????", column
            )
            if df.dtypes[option] != 'object':
                genre = st.radio(
                    "??????",
                    ('???', '??????', '??????'))
        with right:
            if df.dtypes[option]=='object':
                s = st.text_input('??????????????????')
            elif(df.dtypes[option]=='int64'):
                s1=right.number_input('??????',step=1)
                s2=right.number_input('??????',step=1)
            elif(df.dtypes[option]=='float64'):
                s1 = right.number_input('??????', step=0.1)
                s2 = right.number_input('??????', step=0.1)

    elif project_name=="?????????????????????":
        df=pd.read_csv('data/test.csv')
        if st.checkbox('??????'):
            st.write (df)
        st.subheader('????????????')
        left, right = st.columns(2)
        with left:
            column = [i for i in df.columns]
            option = st.selectbox(
                "??????", column
            )
            if df.dtypes[option] != 'object':
                genre = st.radio(
                    "??????",
                    ('???', '??????', '??????'))
        with right:
            if df.dtypes[option] == 'object':
                s = st.text_input('??????????????????')
            elif (df.dtypes[option] == 'int64'):
                s1 = right.number_input('??????', step=1)
                s2 = right.number_input('??????', step=1)
            elif (df.dtypes[option] == 'float64'):
                s1 = right.number_input('??????', step=0.1)
                s2 = right.number_input('??????', step=0.1)


    if st.button('??????'):
        if df.dtypes[option] == 'object':
            result2=df.loc[df[option]==s]
            l=len(result2)
        elif (df.dtypes[option] == 'int64' or df.dtypes[option] == 'float64'):
            if(s1>s2):
                st.warning('????????????')
            else:
                result=df.loc[(df[option] >= s1)&(df[option]<=s2)]
                l=len(result)
            if genre == '??????':
                result2 = result.sort_values(by=[option], ascending=True)
            elif genre == '??????':
                result2 = result.sort_values(by=[option], ascending=False)
            else:
                result2=result
        else:
            st.warning('error')
        st.info("????????????%s?????????" % l)
        st.table(result2)

elif sidebar == "????????????":
    st.title("????????????")
    # ???????????????
    project_name = st.selectbox(
        "???????????????",
        ["???????????????", "????????????"]
    )
    if project_name == "???????????????":
        left, right = st.columns(2)
        with left:
            df=pd.read_csv('data/submission.csv')
            df['Price'] = df['Price'].map(lambda x: '%.2f??????' % x)
            st.write(df)
        with right:
            s = st.text_input('Id')
            if st.button('??????'):
                s=float(s)
                st.table(df.loc[df['Id'] == s])
    elif project_name == "????????????":
        st.subheader('?????????????????????????????????')
        df=pd.read_csv('data/test.csv')
        left,right=st.columns(2)
        with left:
            Region = left.selectbox(
                "Region",
                load_npy('Region'),
                index=1
            )
        with right:
            State = right.selectbox(
                "State",
                load_npy('State'),
                index=0
            )
        left, right = st.columns(2)
        with left:
            Listed_On=left.date_input(
                "Listed On",
                datetime.date(2021, 1, 13))
        with right:
            Last_Sold_On=right.date_input(
                "Last Sold On",
                datetime.date(2017, 6, 30))
        left, right = st.columns(2)
        with left:
            Type = left.multiselect(
                'Type',
                load_npy('Type_devided'),
                ['SingleFamily'])
        with right:
             Heating= right.multiselect(
                'Heating',
                load_npy('Heating_devided'),
                ['Central'])
        left, right = st.columns(2)
        with left:
            Cooling = left.multiselect(
                'Cooling',
                load_npy('Cooling_devided'),
                ['Central Air'])
        with right:
            Parking = right.multiselect(
                'Parking',
                load_npy('Parking_devided'),
                ['Garage', 'Garage - Attached', 'Covered'])
        left, right = st.columns(2)
        with left:
            Bedrooms = left.multiselect(
                'Bedrooms',
                load_npy('Bedrooms_devided'),
                ['3'])
        with right:
            Flooring = right.multiselect(
                'Flooring',
                load_npy('Flooring_devided'),
                ['Wood'])
        with left:
            Appliances_included = left.multiselect(
                'Appliances included',
                load_npy('Appliances included_devided'),
                ['Dishwasher', 'Dryer', 'Garbage disposal', 'Microwave', 'Range / Oven', 'Refrigerator', 'Washer'])
        with right:
            Laundry_features = right.multiselect(
                'Laundry features',
                load_npy('Laundry features_devided'),
                ['In Garage'])
        left, right = st.columns(2)
        with left:
            Parking_features = left.multiselect(
                'Parking features',
                load_npy('Parking features_devided'),
                ['Garage', 'Garage - Attached','Covered'])
        left, right = st.columns(2)
        with left:
            Year_built = st.number_input('Year built',value=2020)
        with right:
            Lot = st.number_input('Lot',value=6098.0,step=0.1)
        left, right = st.columns(2)
        with left:
            Bathrooms = st.number_input('Bathrooms', value=2.0,step=0.1)
        with right:
            Full_bathrooms = st.number_input('Full bathrooms', value=2.0,step=0.1)
        left, right = st.columns(2)
        with left:
            Total_spaces = st.number_input('Total spaces', value=2.0,step=0.1)
        with right:
            Garage_spaces = st.number_input('Garage spaces', value=2.0,step=0.1)
        left, right = st.columns(2)
        with left:
            Elementary_School_Score = st.number_input('Elementary School Score', value=5.0,step=0.1)
        with right:
            Elementary_School_Distance = st.number_input('Elementary School Distance', value=0.3,step=0.1)
        left, right = st.columns(2)
        with left:
            Middle_School_Score = st.number_input('Middle School Score', value=6.0,step=0.1)
        with right:
            Middle_School_Distance = st.number_input('Middle School Distance', value=0.6,step=0.1)
        left, right = st.columns(2)
        with left:
            High_School_Score = st.number_input('High School Score', value=7.0,step=0.1)
        with right:
            High_School_Distance = st.number_input('High School Distance', value=0.8,step=0.1)
        left, right = st.columns(2)
        with left:
            Tax_assessed_value = st.number_input('Tax assessed value', value=510000.0,step=0.1)
        with right:
            Annual_tax_amount = st.number_input('Annual tax amount', value=1395.0,step=0.1)
        left, right = st.columns(2)
        with left:
            Listed_Price = st.number_input('Listed Price', value=799000.0,step=0.1)
        with right:
            Last_Sold_Price = st.number_input('Last Sold Price', value=300000.0,step=0.1)
        left, right = st.columns(2)
        with left:
            Total_interior_livable_area = st.number_input('Total interior livable area', value=1200.0,step=0.1)
        with right:
            Zip = st.number_input('Zip', value=95123)
        if st.button('??????'):
            st.text('??????????????????%.2f??????'%predict())

elif sidebar == "????????????":
    st.title("????????????")
    picture_path='picture'
    image_names = [i for i in os.listdir(picture_path) if not i.startswith('.')]
    project_name = st.selectbox(
        "?????????",
        image_names
    )

    for i in image_names:
        if project_name == i:
            image = Image.open(picture_path+'/'+i)
            st.image(image,
                     caption=i,
                    use_column_width=True)
else:
    st.title("??????????????????")
    st.write("????????????????????????????????????")
    image=Image.open('images/main_page_bg.jpg')
    image1 = Image.open('images/1.jpg')
    st.image(image,
           caption='????????????????????????',
           use_column_width=True)
    st.image(image1,
             caption='????????????',
             use_column_width=True)
