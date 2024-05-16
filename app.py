import streamlit  as st
import joblib
import numpy as np  


scaler=joblib.load('scaler.pkl')
model=joblib.load('model.pkl')

st.title('Customer Car Price Estimator App')

st.divider()

st.write("This app is for getting the estimated price of a car based on the features provided by the user. This can be used to get an idea of the price of a car before buying or selling it.")

age=st.number_input("Enter your age", min_value=18, max_value=90, value=40, step=1)
salary=st.number_input("Enter your salary", min_value=1000, max_value=9999999999, step=5000, value=5000)
netwroth=st.number_input("Enter your net worth", min_value=1000, max_value=9999999999, step=20000, value=100000)

X=[age,salary,netwroth]

calculate_btn=st.button("Calculate Price")
st.divider()
if calculate_btn:
    st.balloons()
    X_2=np.array(X)
    X_array=scaler.transform([X_2])

    prediction=model.predict(X_array)

    st.write(f"The estimated price of the car is: {prediction[0][0]:,.2f}" )
    st.write("This is just an estimate and the actual price may vary.")
else:
    st.write("Please click the calculate button to get the estimated price of the car")