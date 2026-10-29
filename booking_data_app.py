import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
st.title("HOTEL BOOKING CANCELLATION")
data=pd.read_csv(r"C:\Users\sriva\Desktop\Hotel_Booking_Cancellation\ML_REPORT")
data['is_cancelled']=data['is_cancelled'].map({"No":0,"Yes":1})

x=data[['Hotel_Type','Lead_Time_Days','Room_Type','Previous_Bookings', 'Previous_Cancellations', 'Booking_Channel', 'Deposit_Type']]

x['Hotel_Type']=x['Hotel_Type'].map({'Unknown':0,'City Hotel':1, 'Resort Hotel':2})
x['Room_Type']=x['Room_Type'].map({'Unknown':0, 'Standard':1,'Suite':2,'Deluxe':3})
x['Booking_Channel']=x['Booking_Channel'].map({'Offline':0,'Online':1,'Travel Agent':2,'Corporate':3})
x['Deposit_Type']=x['Deposit_Type'].map({'No Deposit':0 ,'Non-Refundable':1,'Refundable':2})

y=data['is_cancelled']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

# model=XGBClassifier(max_depth=3,n_estimator=550,learning_rate=0.50)
model=XGBClassifier(max_depth=3,n_estimators=550,learning_rate=0.50)

model.fit(x_train,y_train)

hotel=st.number_input("enter hotel type Unknown-0,City Hotel-1, Resort Hotel-2",step=1)
stay_days=st.number_input("enter lead time days",step=1)
room_type=st.number_input("enter room type Unknown-0,Standard-1,Suite-2,Deluxe-3",step=1)
Previous_Bookings=st.number_input("enter previous bookings",step=1)
Previous_Cancellations=st.number_input("enter previous cancellations",step=1)
Booking_Channel=st.number_input("enter booking channel Offline-0,Online-1,Travel Agent-2,Corporate-3",step=1)
Deposit_Type=st.number_input("enter deposit type 'No Deposit:0 ,Non-Refundable:1,Refundable:2",step=1)

send_button=st.button("submit")
output=model.predict([[hotel,stay_days,room_type,Previous_Bookings,Previous_Cancellations,Booking_Channel,Deposit_Type]])
if send_button:
    if output[0]==1:
        st.success("cancelled")
    else:
      st.write("not cancelled")