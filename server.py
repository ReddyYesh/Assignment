import streamlit as st
import pickle
model = pickle.load(open('model.pkl','rb'))

def predict(sl):
 prediction = model.predict([[sl]])
 return prediction
def main():
    st.title("HR Analytics - Employee retention")
    html_temp = """
    
    <div style = "background-color:#76D7C4;padding:10px">
    <h2 style = "color:white;text-align:center;">Streamlit HR Analytics - Employee retention</h2>
    <h6 style = "color:white;text-align:center;"> If the Prediction is 0 – Person is  Not looking for job change,  1 – Person is Looking for a job change</h6>
    </div>
   


    """
    if st.button("About"):
       st.text("Output: 0 – Not looking for job change")
       st.text("Output: 1 – Looking for a job change ")

    st.markdown(html_temp,unsafe_allow_html=True)
    sl = st.text_input("Enrollee ID","Type Here")
   

    result = ""
    if st.button("Predict"):
        result = predict(sl)

    st.success('The output is  = {}'.format(result))
    
main()