import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('MyRegressionModel.pkl', 'rb'))
car = pd.read_csv('cleaned_car.csv')

def main():
    st.title('Car Price Prediction')

    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    company = st.selectbox('Select Company', companies)
    car_model = st.selectbox('Select Car Model', car_models)
    year = st.selectbox('Select Year', years)
    fuel_type = st.selectbox('Select Fuel Type', fuel_types)
    driven = st.number_input('Enter Kilometers Driven')

    if st.button('Predict'):
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                   data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))
        st.write("Input Data:")
        st.write(input_data)
        
        prediction = model.predict(input_data)
        st.write("Your car price is")
        st.write(prediction)

        if prediction is not None and len(prediction) > 0:
            st.success(f'Predicted Price: {np.round(prediction[0], 2)}')
        else:
            st.error("Prediction failed.")

if __name__ == '__main__':
    main()
